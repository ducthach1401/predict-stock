"""The daily report (Markdown, HTML, CSV) and the optional notifications (Telegram, e-mail; credentials only from the environment, off by default)."""
from __future__ import annotations

import html
import json
import os
import smtplib
from datetime import date
from email.message import EmailMessage
from pathlib import Path
from typing import Callable

import pandas as pd
import requests
from sqlalchemy import Engine, func, select

from predict_stock.backtest.runner import Setup
from predict_stock.config import PROJECT_ROOT, AppConfig
from predict_stock.db.models import (
    Alert, Instrument, InstrumentSymbolHistory, PaperOrder, PaperPosition, PortfolioSnapshot, Recommendation, RecommendationOutcome,
)
from predict_stock.db.session import session_scope
from predict_stock.paper.alerts import open_alerts


def _sym(engine: Engine) -> dict[int, str]:
    with session_scope(engine) as s:
        rows = s.execute(select(InstrumentSymbolHistory.instrument_id, InstrumentSymbolHistory.symbol, InstrumentSymbolHistory.valid_from).order_by(InstrumentSymbolHistory.valid_from)).all()
    return {i: sym for i, sym, _ in rows}


def collect(engine: Engine, cfg: AppConfig, d: pd.Timestamp, results: dict | None = None) -> dict:
    """Everything the report shows, as plain data (also what the CSV files contain)."""
    pc, sym = cfg.paper.portfolio_code, _sym(engine)
    with session_scope(engine) as s:
        recs = s.execute(select(Recommendation.id, Recommendation.strategy, Recommendation.instrument_id, Recommendation.as_of_date, Recommendation.action, Recommendation.status,
                                Recommendation.entry_price, Recommendation.target_price, Recommendation.stop_loss, Recommendation.valid_until, Recommendation.confidence, Recommendation.flags,
                                Recommendation.card).where(Recommendation.as_of_date == d.date()).order_by(Recommendation.strategy, Recommendation.id)).all()
        statuses = dict(s.execute(select(Recommendation.status, func.count()).where(Recommendation.action == "BUY").group_by(Recommendation.status)).all())
        pos = s.execute(select(PaperPosition).where(PaperPosition.portfolio_code.like(f"{pc}:%"), PaperPosition.status == "open")).scalars().all()
        orders = s.execute(select(PaperOrder).where(PaperOrder.portfolio_code.like(f"{pc}:%"), PaperOrder.placed_date == d.date()).order_by(PaperOrder.id)).scalars().all()
        pending = s.execute(select(PaperOrder).where(PaperOrder.portfolio_code.like(f"{pc}:%"), PaperOrder.status == "pending")).scalars().all()
        snaps = s.execute(select(PortfolioSnapshot).where(PortfolioSnapshot.portfolio_code == pc).order_by(PortfolioSnapshot.snapshot_date)).scalars().all()
        outs = s.execute(select(RecommendationOutcome, Recommendation.strategy, Recommendation.instrument_id, Recommendation.status)
                         .join(Recommendation, Recommendation.id == RecommendationOutcome.recommendation_id).where(RecommendationOutcome.exit_date.is_not(None))).all()
        alerts = [a for a in open_alerts(engine)]
    last = snaps[-1] if snaps else None
    first = snaps[0] if snaps else None
    closed = [{"strategy": st, "symbol": sym.get(iid), "status": stt, "exit_date": str(o.exit_date), "holding_days": o.holding_days, "net_return": o.net_return, "gross_return": o.gross_return} for o, st, iid, stt in outs]
    return {
        "as_of": str(d.date()), "mode": cfg.run.mode, "results": results or {},
        "cards": [{"strategy": r.strategy, "symbol": sym.get(r.instrument_id), "action": r.action, "status": r.status, "entry": float(r.entry_price), "target": float(r.target_price),
                   "stop_or_bear": float(r.stop_loss), "valid_until": str(r.valid_until), "confidence": r.confidence, "flags": r.flags, "rejected": None,
                   "zone": [r.card["entry"]["zone_low"], r.card["entry"]["zone_high"]] if r.card else None} for r in recs],
        "status_counts": statuses,
        "positions": [{"portfolio": p.portfolio_code, "symbol": sym.get(p.instrument_id), "quantity": p.quantity, "avg_cost": float(p.avg_cost), "opened": str(p.opened_date)} for p in pos],
        "orders_today": [{"portfolio": o.portfolio_code, "symbol": sym.get(o.instrument_id), "side": o.side, "type": o.order_type, "qty": o.quantity, "status": o.status, "fill_price": o.fill_price} for o in orders],
        "pending_orders": [{"portfolio": o.portfolio_code, "symbol": sym.get(o.instrument_id), "side": o.side, "type": o.order_type, "qty": o.quantity, "limit": None if o.limit_price is None else float(o.limit_price)} for o in pending],
        "equity": None if last is None else {"date": str(last.snapshot_date), "equity": float(last.equity), "cash": float(last.cash), "market_value": float(last.market_value), "metrics": last.metrics,
                                            "since_start": float(last.equity / first.equity - 1)},
        "equity_curve": [{"date": str(x.snapshot_date), "equity": float(x.equity), "cash": float(x.cash)} for x in snaps],
        "closed": closed,
        "alerts": [{"id": a.id, "severity": a.severity, "category": a.category, "message": a.message, "created": str(a.created_at)} for a in alerts[:30]],
    }


def render_md(c: dict) -> str:
    L = [f"# Paper trading report — {c['as_of']}", f"Mode **{c['mode']}** (no real order is ever placed). All figures are simulated with the same rules as the backtest.\n"]
    res = c["results"]
    steps = {k: v for k, v in res.items() if isinstance(v, dict) and "seconds" in v}
    if steps:
        L.append("## Jobs\n")
        L.append("| step | ok | seconds | note |\n| --- | --- | --- | --- |")
        for k, v in steps.items():
            r = v.get("result") if isinstance(v.get("result"), dict) else {}
            note = v.get("error") or (r.get("skipped") if r else "") or ""
            L.append(f"| {k} | {'yes' if v['ok'] else '**FAILED**'} | {v['seconds']} | {note} |")
        L.append("")
    e = c["equity"]
    if e:
        m = e["metrics"] or {}
        L.append("## Portfolio\n")
        L.append(f"Equity {e['equity']:,.0f} VND on {e['date']} ({e['since_start'] * 100:+.2f}% since the start of the record), cash {e['cash']:,.0f}, invested {m.get('exposure', 0) * 100:.0f}%, "
                 f"drawdown from peak {m.get('drawdown', 0) * 100:.1f}%." + (f" **Kill-switch fired: {', '.join(m['kill_events'])}.**" if m.get("kill_events") else ""))
        L.append("")
    L.append(f"## Cards for the next session ({c['as_of']})\n")
    if c["cards"]:
        L.append("| strategy | symbol | action | status | entry zone | target | stop / bear ref. | valid until | flags |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for x in c["cards"]:
            z = f"{x['zone'][0]:,.0f}–{x['zone'][1]:,.0f}" if x["zone"] else ""
            L.append(f"| {x['strategy']} | {x['symbol']} | {x['action']} | {x['status']} | {z} | {x['target']:,.0f} | {x['stop_or_bear']:,.0f} | {x['valid_until']} | {json.dumps(x['flags'], ensure_ascii=False) if x['flags'] else ''} |")
    else:
        L.append("No card was issued for this session.")
    cards_res = (res.get("cards") or {}).get("result") or {}
    if cards_res.get("rejected"):
        L.append("\nNot issued (with the reason):\n")
        L += [f"* {r}" for r in cards_res["rejected"]]
    L.append("")
    L.append("## Recommendations by status\n")
    L.append(", ".join(f"{k}: {v}" for k, v in sorted(c["status_counts"].items())) or "none")
    L.append("")
    L.append("## Orders\n")
    L.append("| portfolio | symbol | side | type | qty | status | fill |\n| --- | --- | --- | --- | --- | --- | --- |")
    for o in c["orders_today"]:
        L.append(f"| {o['portfolio']} | {o['symbol']} | {o['side']} | {o['type']} | {o['qty']:,} | {o['status']} | {o['fill_price'] or ''} |")
    L.append(f"\nPending orders for the next session: {len(c['pending_orders'])}.\n")
    L.append("## Open positions (per sleeve)\n")
    L.append("| sleeve | symbol | quantity | average cost | opened |\n| --- | --- | --- | --- | --- |")
    for p in sorted(c["positions"], key=lambda p: (p["portfolio"], p["symbol"] or "")):
        L.append(f"| {p['portfolio']} | {p['symbol']} | {p['quantity']:,} | {p['avg_cost']:,.0f} | {p['opened']} |")
    L.append("")
    if c["closed"]:
        d = pd.DataFrame(c["closed"])
        L.append(f"## Closed recommendations\n\n{len(d)} closed; mean net return {d['net_return'].mean() * 100:.2f}%, {(d['net_return'] > 0).mean() * 100:.0f}% positive.\n")
    L.append("## Alerts (open)\n")
    L += [f"* **{a['severity']}** [{a['category']}] {a['message']} ({a['created']})" for a in c["alerts"]] or ["none"]
    return "\n".join(L).rstrip() + "\n"


def render_html(md: str) -> str:
    """A small, dependency-free Markdown -> HTML (headings, tables, bullets, bold)."""
    out, table = [], []

    def flush():
        if table:
            head, *_, = table
            rows = [r for r in table[2:]]
            cells = lambda line: [c.strip() for c in line.strip().strip("|").split("|")]
            out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells(head)) + "</tr></thead><tbody>" +
                       "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells(r)) + "</tr>" for r in rows) + "</tbody></table>")
            table.clear()

    def inline(t: str) -> str:
        t = html.escape(t)
        import re
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)

    for line in md.splitlines():
        if line.startswith("|"):
            table.append(line)
            continue
        flush()
        if line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("* "):
            out.append(f"<li>{inline(line[2:])}</li>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    flush()
    css = "body{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#0b0b0b}table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ddd;padding:4px 8px;font-size:13px}th{background:#f3f3f1}"
    return f"<!doctype html><html><head><meta charset='utf-8'><title>Paper report</title><style>{css}</style></head><body>{''.join(out)}</body></html>"


def write_files(cfg: AppConfig, c: dict) -> dict[str, str]:
    base = PROJECT_ROOT / cfg.paper.report_dir / c["as_of"]
    base.mkdir(parents=True, exist_ok=True)
    files = {}
    md = render_md(c)
    fmt = cfg.paper.report_formats
    if "md" in fmt:
        (base / "report.md").write_text(md, encoding="utf-8")
        files["md"] = str(base / "report.md")
    if "html" in fmt:
        (base / "report.html").write_text(render_html(md), encoding="utf-8")
        files["html"] = str(base / "report.html")
    if "csv" in fmt:
        for name, rows in (("cards", c["cards"]), ("positions", c["positions"]), ("orders_today", c["orders_today"]), ("pending_orders", c["pending_orders"]), ("equity_curve", c["equity_curve"]),
                           ("closed", c["closed"]), ("alerts", c["alerts"])):
            pd.DataFrame(rows).to_csv(base / f"{name}.csv", index=False)
            files[f"csv:{name}"] = str(base / f"{name}.csv")
    return files


# ---- notifications ------------------------------------------------------------------------------------------------------------------------------------
def summary_text(c: dict) -> str:
    e = c["equity"]
    buys = [x for x in c["cards"] if x["action"] == "BUY"]
    bad = [a for a in c["alerts"] if a["severity"] in ("error", "critical")]
    failed = [k for k, v in c["results"].items() if isinstance(v, dict) and "seconds" in v and not v["ok"]]
    return (f"Paper {c['as_of']}: {len(buys)} BUY / {sum(x['action'] == 'WATCH' for x in c['cards'])} WATCH cards; "
            + (f"equity {e['equity']:,.0f} ({e['since_start'] * 100:+.2f}%); " if e else "") + f"pending orders {len(c['pending_orders'])}; open alerts {len(c['alerts'])}"
            + (f"; FAILED steps: {', '.join(failed)}" if failed else "") + (f"; {len(bad)} error alert(s)" if bad else "") + ".")


def send_telegram(text: str, env=os.environ) -> None:
    token, chat = env.get("TELEGRAM_BOT_TOKEN"), env.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        raise RuntimeError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set")
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat, "text": text[:4000]}, timeout=20)
    r.raise_for_status()


def send_email(subject: str, text: str, attachment: Path | None, env=os.environ) -> None:
    need = ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM", "SMTP_TO")
    miss = [k for k in need if not env.get(k)]
    if miss:
        raise RuntimeError(f"missing e-mail settings: {', '.join(miss)}")
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, env["SMTP_FROM"], env["SMTP_TO"]
    msg.set_content(text)
    if attachment is not None and attachment.exists():
        msg.add_attachment(attachment.read_text(encoding="utf-8"), subtype="html", filename=attachment.name)
    with smtplib.SMTP(env["SMTP_HOST"], int(env.get("SMTP_PORT", "587")), timeout=30) as smtp:
        smtp.starttls()
        smtp.login(env["SMTP_USER"], env["SMTP_PASSWORD"])
        smtp.send_message(msg)


def notify(engine: Engine, cfg: AppConfig, c: dict, files: dict, *, telegram: Callable[[str], None] | None = None, email: Callable[[str, str, Path | None], None] | None = None) -> dict:
    """Sends the summary where configured. A failure to notify is an alert, never an exception of the day's job."""
    from predict_stock.paper.alerts import raise_alert
    n = cfg.paper.notify
    if n.only_on_alerts_or_cards and not (c["alerts"] or c["cards"]):
        return {"sent": []}
    text, sent = summary_text(c), []
    for name, on, fn in (("telegram", n.telegram, telegram or send_telegram), ("email", n.email, email or (lambda s, t, a: send_email(s, t, a)))):
        if not on:
            continue
        try:
            fn(text) if name == "telegram" else fn(f"Paper report {c['as_of']}", text, Path(files["html"]) if "html" in files else None)
            sent.append(name)
        except Exception as exc:                                                     # noqa: BLE001
            raise_alert(engine, "warn", "notify_failed", f"{name} notification failed: {type(exc).__name__}: {exc}")
    return {"sent": sent}


def run_report(engine: Engine, cfg: AppConfig, setup: Setup, d: pd.Timestamp, results: dict) -> dict:
    c = collect(engine, cfg, d, results)
    files = write_files(cfg, c)
    return {"files": files, "notified": notify(engine, cfg, c, files)["sent"]}
