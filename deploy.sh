#!/usr/bin/env bash
# predict-stock in Docker: one file to deploy and operate it. Paper trading only: nothing here places a real order.
#
#   bash deploy.sh                      deploy (idempotent): .env, image, MySQL, schema; on an EMPTY database the first-time set-up (data + models, takes a while);
#                                       then the scheduler container (daily job 16:30 Asia/Ho_Chi_Minh, backup 17:30, restore test Sundays)
#   bash deploy.sh --bundle FILE        the same, but the empty database is filled from a bundle made by `export` (moves a trained system to a new machine, no retraining)
#   bash deploy.sh export [FILE]        bundle = a database dump + artifacts (models, datasets) + config + universe files
#   bash deploy.sh status               containers, paper record, alerts, schedule
#   bash deploy.sh run-now [YYYY-MM-DD] run the daily job now (idempotent)
#   bash deploy.sh logs                 follow the scheduler
#   bash deploy.sh cli <args>           any `python -m predict_stock <args>` inside the container, e.g.  bash deploy.sh cli lifecycle status
#   bash deploy.sh shell                a shell in the container
#   bash deploy.sh stop                 stop the containers (the database volume is kept)
set -euo pipefail
cd "$(dirname "$0")"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- docker and compose -------------------------------------------------------------------------------------------------------------------------
command -v docker >/dev/null || die "docker is not installed (https://docs.docker.com/engine/install/)"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is it running? are you in the 'docker' group? otherwise run this script with sudo)"
if docker compose version >/dev/null 2>&1; then C="docker compose"
elif command -v docker-compose >/dev/null; then C="docker-compose"
else die "neither 'docker compose' nor 'docker-compose' is available"; fi

# ---- .env: created once, never overwritten ------------------------------------------------------------------------------------------------------
rand() { if command -v openssl >/dev/null; then openssl rand -hex 16; else head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n'; fi; }
make_env() {
  [ -f .env ] && return 0
  [ -f .env.example ] || die ".env.example is missing"
  say "creating .env with generated passwords (mode 600, git-ignored)"
  sed -e "s/change_me_root/$(rand)/" -e "s/change_me_app/$(rand)/" -e "s/change_me_migrator/$(rand)/" .env.example > .env
  chmod 600 .env
}
ensure_env_key() { grep -q "^$1=" .env || printf '%s=%s\n' "$1" "$2" >> .env; }   # only adds what is missing
make_env
ensure_env_key APP_UID "$(id -u)"
ensure_env_key APP_GID "$(id -g)"
set -a; . ./.env; set +a
MYSQL_CONTAINER="${MYSQL_CONTAINER:-predict-stock-mysql}"

# host directories the containers write to: created here so they belong to you, not to root
mkdir -p artifacts reports backups logs docs data config

app()  { $C run --rm -T app "$@"; }                      # one-off command in a fresh app container
cli()  { app python -m predict_stock "$@"; }

sql() {                                                   # a query against the real database, via the MySQL container (prints nothing on error)
  $C exec -T mysql sh -c 'MYSQL_PWD="$MYSQL_ROOT_PASSWORD" mysql -uroot -N -B "$MYSQL_DATABASE" -e "$0"' "$1" 2>/dev/null || true
}

wait_for_mysql() {
  say "waiting for MySQL"
  for _ in $(seq 1 90); do
    [ "$(docker inspect -f '{{.State.Health.Status}}' "$MYSQL_CONTAINER" 2>/dev/null || true)" = healthy ] && return 0
    sleep 2
  done
  die "MySQL did not become healthy in 3 minutes: $C logs mysql"
}

# ---- commands that do not deploy ----------------------------------------------------------------------------------------------------------------
cmd="${1:-up}"
case "$cmd" in
  status)
    $C ps
    cli paper status || true
    cli paper scheduler --show || true
    exit 0 ;;
  logs)    exec $C logs -f --tail=100 app ;;
  stop)    $C stop; exit 0 ;;
  shell)   exec $C run --rm app bash ;;
  cli)     shift; cli "$@"; exit $? ;;
  run-now) shift; if [ $# -gt 0 ]; then cli paper run --as-of "$1"; else cli paper run; fi; exit $? ;;
  export)
    shift
    out="${1:-predict-stock-bundle-$(date +%Y%m%d-%H%M).tar.gz}"
    $C build app
    $C up -d mysql; wait_for_mysql
    say "database dump"
    cli paper backup
    dump="$(ls -t backups/predict_stock_*.sql.gz | head -1)"
    tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
    mkdir -p "$tmp/bundle"
    cp "$dump" "$tmp/bundle/db.sql.gz"
    cp -r artifacts config data "$tmp/bundle/"
    ( cd "$tmp/bundle" && { echo "created $(date -u +%FT%TZ)"; echo "git $(git -C "$OLDPWD" rev-parse HEAD 2>/dev/null || echo unknown)"; echo "source dump $(basename "$dump")"; } > MANIFEST.txt \
      && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
    tar -C "$tmp" -czf "$out" bundle
    say "bundle: $out ($(du -h "$out" | cut -f1)). On the new machine:  bash deploy.sh --bundle $out"
    exit 0 ;;
  up|--bundle) ;;
  *) die "unknown command '$cmd' (see the header of deploy.sh)" ;;
esac
bundle=""
if [ "$cmd" = "--bundle" ]; then bundle="${2:-}"; [ -f "$bundle" ] || die "usage: bash deploy.sh --bundle FILE"; fi

# ---- deploy -------------------------------------------------------------------------------------------------------------------------------------
say "building the image"
$C build app
say "starting MySQL"
$C up -d mysql
wait_for_mysql

trained="$(sql 'SELECT COUNT(*) FROM models' | tr -d '[:space:]')"
if [ "${trained:-0}" -gt 0 ] 2>/dev/null; then
  say "the database already holds models: schema check only"
  app alembic upgrade head
elif [ -n "$bundle" ]; then
  say "restoring the bundle $bundle"
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  tar -C "$tmp" -xzf "$bundle"
  ( cd "$tmp/bundle" && sha256sum -c --quiet SHA256SUMS ) || die "the bundle is damaged (checksum mismatch)"
  cp -r "$tmp/bundle/artifacts/." artifacts/
  gunzip -c "$tmp/bundle/db.sql.gz" | $C exec -T mysql sh -c 'MYSQL_PWD="$MYSQL_ROOT_PASSWORD" mysql -uroot "$MYSQL_DATABASE"'
  app alembic upgrade head
  cli lifecycle init-champions
else
  say "empty database: first-time set-up (schema, universe, market data, features, baselines, SWING and INVEST models). This takes a while; it is safe to interrupt and run again"
  app bash docker/bootstrap.sh
fi

say "starting the scheduler"
$C up -d app
sleep 3
$C ps
cli paper status || true
cat <<EOF

Deployed. Paper trading only. The scheduler runs the daily job at 16:30 (Asia/Ho_Chi_Minh) Mon-Fri, a backup at 17:30 and a restore test on Sundays.
  bash deploy.sh status          bash deploy.sh run-now          bash deploy.sh logs          bash deploy.sh export
Reports:  reports/paper/<date>/report.md      Logs:  logs/      Backups:  backups/
EOF
