#!/usr/bin/env python3
"""
AI Trading Agent — Full Watchlist Monitor
------------------------------------------
Monitors all 5 watchlist tickers autonomously:
  Large caps : AAPL, NVDA, MSFT
  Small caps : SMCI, PLTR

On each cycle it:
  1. Fetches portfolio + all open positions
  2. Checks every open order for fill status
  3. Monitors ALL positions against their individual stop-loss levels
  4. Scans news for EACH watchlist ticker independently
  5. For each qualifying signal: review → execute → notify
  6. Persists updated state to state.json for the next cycle

Setup:
  pip install anthropic pytz

Required environment variables:
  ANTHROPIC_API_KEY        — your Anthropic API key

Optional (auto-managed by script via state.json):
  STATE_FILE               — path to state JSON (default: ./agent_state.json)

Run on a schedule (every 5 min during market hours):
  */5 9-16 * * 1-5 /usr/bin/python3 /path/to/trading_agent_full.py

Or deploy to AWS Lambda / GitHub Actions (see SETUP.md).
"""

import os
import json
import logging
import datetime
import pytz
import anthropic
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

ACCOUNT_NUMBER = "426374179"          # Agentic account ••••4179

# Watchlist — grouped for logging clarity
LARGE_CAPS  = ["AAPL", "NVDA", "MSFT"]
SMALL_CAPS  = ["SMCI", "PLTR"]
WATCHLIST   = LARGE_CAPS + SMALL_CAPS

# Risk parameters (from your rule set)
MAX_ORDER_CAP    = 100.0   # $ hard cap per single order
MAX_POSITION_CAP = 100.0   # $ hard cap per ticker position
STOP_LOSS_PCT    = 0.90    # sell if price drops to entry × 0.90
MIN_NEWS_SOURCES = 2       # independent sources required to trigger a BUY

ROBINHOOD_MCP = "https://agent.robinhood.com/mcp/trading"
MARKET_TZ     = pytz.timezone("America/New_York")
MARKET_OPEN   = datetime.time(9, 30)
MARKET_CLOSE  = datetime.time(16, 0)

STATE_FILE = Path(os.environ.get("STATE_FILE", "./agent_state.json"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("trading_agent")


# ── State management ──────────────────────────────────────────────────────────

DEFAULT_STATE = {
    "stop_loss_map": {},    # { "SMCI": 30.20, "AAPL": 261.45, ... }
    "open_order_ids": [],   # [ "uuid1", "uuid2", ... ]
    "positions": {},        # { "SMCI": { "qty": 2, "avg_cost": 33.56 }, ... }
    "last_run": None,
}

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception as e:
            log.warning(f"Could not load state file: {e} — using defaults.")
    return dict(DEFAULT_STATE)

def save_state(state: dict):
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2))
        log.info(f"State saved to {STATE_FILE}")
    except Exception as e:
        log.error(f"Could not save state: {e}")


# ── Market hours guard ────────────────────────────────────────────────────────

def is_market_open() -> bool:
    now_et = datetime.datetime.now(MARKET_TZ)
    if now_et.weekday() >= 5:
        return False
    return MARKET_OPEN <= now_et.time() <= MARKET_CLOSE


# ── Anthropic + Robinhood MCP call ───────────────────────────────────────────

def call_agent(task_prompt: str, system_prompt: str) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=api_key)

    response = client.beta.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8096,
        system=system_prompt,
        messages=[{"role": "user", "content": task_prompt}],
        mcp_servers=[{"type": "url", "url": ROBINHOOD_MCP, "name": "robinhood"}],
        betas=["mcp-client-2025-04-04"],
    )

    return response.model_dump()


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are an autonomous AI trading agent connected to a live Robinhood brokerage account.
You execute trades using REAL money. Be disciplined, precise, and conservative.

ACCOUNT: {ACCOUNT_NUMBER} — use this account_number for ALL tool calls (agentic_allowed=true).

WATCHLIST:
  Large caps : {", ".join(LARGE_CAPS)}
  Small caps : {", ".join(SMALL_CAPS)}

CORE PHILOSOPHY:
- Rely exclusively on measurable sentiment from specific, confirmed news events.
- Do NOT use technical analysis, chart patterns, or macroeconomic forecasts.
- If any condition is ambiguous or unmet → do NOT trade; log as PENDING.

STRICT RISK RULES (never violate):
- Hard cap: ${MAX_ORDER_CAP:.2f} max per single order
- Hard cap: ${MAX_POSITION_CAP:.2f} max position size per ticker
- Stop-loss: entry_price × {STOP_LOSS_PCT} — place SELL immediately if breached
- Require ≥{MIN_NEWS_SOURCES} independent confirming sources for any BUY signal
- ALWAYS call review_equity_order before place_equity_order; abort if any warnings returned
- Only trade 9:30 AM – 4:00 PM ET on weekdays (market hours only)
- Only add a ticker to the watchlist if tied to a high-conviction news signal

SEQUENTIAL EXECUTION PIPELINE (for every trade opportunity, in order):
1. SCAN   — Parse news for each watchlist ticker individually
2. CHECK  — Verify ≥{MIN_NEWS_SOURCES} independent sources confirm the same catalyst
3. REVIEW — Call review_equity_order; verify zero warnings/errors
4. EXECUTE — Call place_equity_order with limit parameters
5. NOTIFY — Generate the notification schema (included in your JSON output)

DYNAMIC TICKER ADDITIONS:
- You may add a ticker outside the base watchlist ONLY if it has a high-conviction,
  multi-source news catalyst directly tied to a watchlist company (e.g. a supplier,
  key customer, or direct competitor). Log the reason clearly.

OUTPUT FORMAT — respond ONLY with valid JSON (no markdown, no preamble):
{{
  "timestamp": "<ISO8601>",
  "account": "{ACCOUNT_NUMBER}",
  "portfolio_value": <float>,
  "cash": <float>,
  "positions": [
    {{
      "symbol": "",
      "quantity": <float>,
      "avg_cost": <float>,
      "current_price": <float>,
      "stop_loss": <float>,
      "pnl_pct": <float>,
      "stop_triggered": <bool>
    }}
  ],
  "orders_checked": [
    {{
      "order_id": "",
      "symbol": "",
      "state": "",
      "filled_qty": <float>
    }}
  ],
  "ticker_scans": [
    {{
      "symbol": "",
      "catalyst": "",
      "sources": [],
      "source_count": <int>,
      "signal": "BUY | SELL | NONE | PENDING",
      "blocked_reason": ""
    }}
  ],
  "actions_taken": [
    {{
      "type": "BUY | SELL | STOP_LOSS",
      "symbol": "",
      "quantity": <float>,
      "price": <float>,
      "total": <float>,
      "stop_loss_price": <float>,
      "reason": "",
      "sources": [],
      "order_id": ""
    }}
  ],
  "signals_pending": [
    {{
      "symbol": "",
      "catalyst": "",
      "reason": ""
    }}
  ],
  "dynamic_additions": [
    {{
      "symbol": "",
      "reason": ""
    }}
  ],
  "errors": []
}}
""".strip()


# ── Task builder ──────────────────────────────────────────────────────────────

def build_task(state: dict) -> str:
    stop_loss_map  = state.get("stop_loss_map", {})
    open_order_ids = state.get("open_order_ids", [])

    parts = [
        f"Run a full trading cycle on account {ACCOUNT_NUMBER}.",
        "",
        "━━ STEP 1 — PORTFOLIO ━━",
        "Fetch current portfolio and all open equity positions.",
        "",
    ]

    if open_order_ids:
        parts += [
            "━━ STEP 2 — ORDER STATUS ━━",
            f"Check fill status for these open order IDs: {json.dumps(open_order_ids)}",
            "Log the state and filled quantity for each order.",
            "",
        ]

    parts += [
        "━━ STEP 3 — STOP-LOSS MONITOR ━━",
        "For every open position, fetch the current quote and compare against its stop-loss.",
        "Stop-loss formula: entry_price × 0.90.",
    ]
    if stop_loss_map:
        parts.append("Current stop-loss levels from state:")
        for sym, sl in stop_loss_map.items():
            parts.append(f"  {sym}: ${sl:.2f}")
    else:
        parts.append("No stop-loss levels in state yet — derive from open positions if any.")
    parts.append("If any position breaches its stop-loss → review_equity_order then place_equity_order (SELL, full quantity).")
    parts.append("")

    parts += [
        "━━ STEP 4 — NEWS SCAN (scan each ticker independently) ━━",
    ]
    for sym in WATCHLIST:
        parts.append(f"  {sym}: search for recent news and evaluate for a BUY or SELL signal.")
    parts += [
        f"For each ticker: require ≥{MIN_NEWS_SOURCES} independent confirming sources for the same catalyst.",
        f"Position cap: ${MAX_POSITION_CAP:.2f} per ticker. Do not buy if already at cap.",
        "For each qualifying BUY signal: call review_equity_order (abort on warnings), then place_equity_order.",
        "For each qualifying SELL signal on a held position: review then sell.",
        "",
        "━━ STEP 5 — REPORT ━━",
        "Return the full structured JSON report as defined in the system prompt.",
        "Include ticker_scans for EVERY watchlist ticker, even those with no signal.",
    ]

    return "\n".join(parts)


# ── Response parser ───────────────────────────────────────────────────────────

def parse_report(response: dict) -> dict:
    for block in response.get("content", []):
        if block.get("type") == "text":
            text = block["text"].strip()
            # Strip markdown fences
            if "```" in text:
                parts = text.split("```")
                for part in parts:
                    candidate = part.lstrip("json").strip()
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
            else:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    pass
    return {"errors": ["Could not parse agent JSON report from response."]}


# ── Notification formatter ────────────────────────────────────────────────────

def format_notification(action: dict) -> str:
    action_type = action.get("type", "")
    symbol      = action.get("symbol", "")
    qty         = action.get("quantity", 0)
    price       = action.get("price", 0.0)
    total       = action.get("total", 0.0)
    reason      = action.get("reason", "")
    sources     = action.get("sources", [])
    stop        = action.get("stop_loss_price") or round(price * (1 - STOP_LOSS_PCT), 2)

    lines = [
        f"[{action_type}] {symbol} · {qty} shares @ ${price:.2f} · Total: ${total:.2f}",
        f"Reason: {reason}",
    ]
    if sources:
        lines.append(f"Sources: {', '.join(sources)}")
    if action_type == "BUY":
        lines.append(f"Stop-loss set at: ${stop:.2f}")
    return "\n".join(lines)


# ── State updater ─────────────────────────────────────────────────────────────

def update_state(state: dict, report: dict) -> dict:
    """Merge the agent's report back into persistent state."""
    updated = dict(state)

    # Update stop-loss map from actions
    stop_map = dict(updated.get("stop_loss_map", {}))
    positions = dict(updated.get("positions", {}))
    order_ids = list(updated.get("open_order_ids", []))

    for action in report.get("actions_taken", []):
        sym  = action.get("symbol", "")
        atype = action.get("type", "")
        price = action.get("price", 0.0)
        qty   = action.get("quantity", 0.0)
        oid   = action.get("order_id", "")

        if atype == "BUY":
            stop_map[sym] = round(price * STOP_LOSS_PCT, 2)
            positions[sym] = {"qty": qty, "avg_cost": price}
            if oid and oid not in order_ids:
                order_ids.append(oid)

        elif atype in ("SELL", "STOP_LOSS"):
            stop_map.pop(sym, None)
            positions.pop(sym, None)

    # Remove filled/cancelled orders from tracking
    filled_states = {"filled", "cancelled", "rejected", "failed", "voided"}
    checked_ids   = {
        o.get("order_id"): o.get("state", "")
        for o in report.get("orders_checked", [])
    }
    order_ids = [
        oid for oid in order_ids
        if checked_ids.get(oid, "open") not in filled_states
    ]

    updated["stop_loss_map"]  = stop_map
    updated["positions"]      = positions
    updated["open_order_ids"] = order_ids
    updated["last_run"]       = datetime.datetime.now(MARKET_TZ).isoformat()

    return updated


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 65)
    log.info("Full-watchlist trading agent — cycle starting")

    state = load_state()
    log.info(f"Loaded state: stop_losses={list(state['stop_loss_map'].keys())}  "
             f"open_orders={len(state['open_order_ids'])}")

    # Market hours gate
    if not is_market_open():
        now_et = datetime.datetime.now(MARKET_TZ)
        log.info(f"Market closed ({now_et.strftime('%A %H:%M ET')}) — skipping cycle.")
        return

    task = build_task(state)
    log.info("Calling Claude agent with Robinhood MCP...")
    log.info(f"Watchlist this cycle: {', '.join(WATCHLIST)}")

    try:
        response = call_agent(task, SYSTEM_PROMPT)
    except Exception as e:
        log.error(f"Agent call failed: {e}")
        return

    report = parse_report(response)

    # ── Log errors ──
    for err in report.get("errors", []):
        log.error(f"Agent error: {err}")

    # ── Log portfolio ──
    log.info(f"Portfolio: ${report.get('portfolio_value', '?')}  "
             f"Cash: ${report.get('cash', '?')}")

    # ── Log positions ──
    for pos in report.get("positions", []):
        triggered = " ⚠ STOP TRIGGERED" if pos.get("stop_triggered") else ""
        log.info(
            f"  {pos['symbol']:5s}  {pos['quantity']}sh  "
            f"avg ${pos['avg_cost']:.2f}  now ${pos['current_price']:.2f}  "
            f"stop ${pos['stop_loss']:.2f}  PnL {pos['pnl_pct']:+.1f}%{triggered}"
        )

    # ── Log order status ──
    for order in report.get("orders_checked", []):
        log.info(
            f"  Order {order['order_id'][:8]}…  {order['symbol']}  "
            f"state={order['state']}  filled={order['filled_qty']}"
        )

    # ── Log ticker scans ──
    log.info("Ticker scan results:")
    for scan in report.get("ticker_scans", []):
        signal  = scan.get("signal", "NONE")
        sources = scan.get("source_count", 0)
        cat     = scan.get("catalyst", "—")
        blocked = scan.get("blocked_reason", "")
        flag    = "✅" if signal in ("BUY", "SELL") else ("⏳" if signal == "PENDING" else "—")
        log.info(
            f"  {flag} {scan['symbol']:5s}  signal={signal:7s}  "
            f"sources={sources}  catalyst={cat[:60]}"
            + (f"  blocked={blocked}" if blocked else "")
        )

    # ── Log dynamic additions ──
    for add in report.get("dynamic_additions", []):
        log.info(f"  DYNAMIC ADD: {add['symbol']} — {add['reason']}")

    # ── Log actions ──
    for action in report.get("actions_taken", []):
        note = format_notification(action)
        log.info("ACTION TAKEN:\n" + note)

    # ── Log pending signals ──
    for sig in report.get("signals_pending", []):
        log.info(f"  PENDING: {sig['symbol']} — {sig.get('catalyst', sig.get('reason', ''))}")

    # ── Update and save state ──
    new_state = update_state(state, report)
    save_state(new_state)

    log.info("Cycle complete.")
    log.info(f"Active stop-losses: {new_state['stop_loss_map']}")
    log.info(f"Tracking {len(new_state['open_order_ids'])} open order(s).")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
