# Deploying with Docker

One file does everything: `bash deploy.sh`. Paper trading only: nothing in the image can place a real order.

Needs: Docker (with `docker compose` or `docker-compose`), internet access to the DNSE public endpoint, about 2 GB of disk.

## Commands

```
bash deploy.sh                    # deploy: .env, image, MySQL, schema, (first time: data + models), scheduler
bash deploy.sh --bundle FILE      # same, but an empty database is filled from a bundle made by `export` (no retraining)
bash deploy.sh export [FILE]      # bundle = database dump + artifacts (models, datasets) + config + universe files
bash deploy.sh status             # containers, paper record, alerts, schedule
bash deploy.sh run-now [DATE]     # run the daily job now (idempotent)
bash deploy.sh logs               # follow the scheduler
bash deploy.sh cli <args>         # any `python -m predict_stock <args>` inside the container, e.g. cli lifecycle status
bash deploy.sh shell              # a shell in the container
bash deploy.sh stop               # stop the containers (the database volume is kept)
```

`deploy.sh` is idempotent: running it again rebuilds the image (cached when nothing changed), applies pending migrations and keeps the scheduler up. To update the code: pull, then `bash deploy.sh`.

## What runs

| container | what |
|---|---|
| `mysql` (MySQL 8.0, volume `predict_stock_mysql`, published on `127.0.0.1:3307` only) | the database |
| `app` (scheduler, `python -m predict_stock paper scheduler`) | daily job 16:30 Asia/Ho_Chi_Minh Mon–Fri, backup 17:30 every day (14 daily + 8 weekly kept), restore test Sundays 03:00. A container that was down catches up when it starts (the job is idempotent); a failed job is retried every 30 minutes, four times a day at most. A healthcheck watches the loop's heartbeat |

Files you can read and edit on the host (bind mounts): `config/` (read-only in the container), `data/` (universe files), `artifacts/` (models, datasets), `reports/paper/<date>/report.md`, `backups/`, `logs/`, `docs/`.

Secrets: `.env` is created once by `deploy.sh` with random passwords (mode 600, git-ignored, never baked into the image, never overwritten). To change the schedule or the paper settings edit `config/default.yaml` and `docker-compose restart app`.

## The first run on an empty database

Two ways, decided by whether you pass `--bundle`:

* **`--bundle`** (recommended when you already have a system you want to keep): the dump, the model files and the datasets are restored exactly, including the paper record, recommendations and alerts. Seconds to minutes. This is how to move the system to another machine: `bash deploy.sh export` on the old one, copy the file, `bash deploy.sh --bundle FILE` on the new one.
* **from zero** (no bundle): `docker/bootstrap.sh` runs schema → universe LARGE50 → market data from DNSE → features and datasets → baselines → SWING and INVEST training → first champions → `paper init`. It trains **new** models on the data of that day, so the result is not the same set of models as an existing system; it also takes a long time.

If the database already holds models, `deploy.sh` only applies migrations and starts the scheduler.

## What was verified (2026-09-21, Docker 29.1, docker-compose 1.29)

* Export from the real system, restore into an isolated second install (own network, volume and port): the restored install shows the same paper record start, 24 recommendations and the 3 alerts; `paper run --as-of 2026-09-18`, `lifecycle monitor` and the **restore test** (34 tables, 267,892 rows, 0 differences) all pass inside the container.
* The scheduler container came up, caught up the day's jobs by itself (daily job and backup, both `ok`) and reports `healthy`.
* The first restore test failed for a real reason (the MySQL client needed `libncurses6`); the first image build silently installed no Python dependencies (the legacy builder of docker-compose 1.x ignores here-documents). Both are fixed and the Dockerfile no longer uses here-documents.
* **From zero** (empty database, no bundle, own network and port): `bash deploy.sh` finished with exit 0. Bootstrap took 13 minutes on this machine (data 1.5, baselines 2.5, SWING 5.5, INVEST 3); it registered the models, made them the first champions (SWING trained until 2025-09-05, INVEST B1 until 2025-06-20, B2 until 2025-03-19), initialised the paper record at 2026-09-21, the scheduler came up healthy and ran that day's daily job by itself (11 steps ok, 5 pending recommendations). These are **new** models trained today; I did not compare their quality with the existing ones.

## Reproducibility

`requirements.lock` is the `pip freeze` of the environment the models were built in; the image installs exactly those versions (pandas, numpy and LightGBM change numbers). After changing dependencies regenerate it (`.venv/bin/pip freeze --exclude-editable > requirements.lock`) and rebuild. A test checks that every declared dependency is pinned.

## Limits

* Backups are written to `./backups` on the same machine; copy them (and bundles) elsewhere.
* Telegram / e-mail notifications stay off unless you enable them in the config and add their credentials to `.env`.
* One scheduler container is assumed. Manual `deploy.sh run-now` while the scheduler is running the same job is safe (idempotent) but not locked.
* docker-compose 1.x with a recent Docker Engine is old but worked here; `docker compose` (v2) is preferred when available and is detected automatically.
