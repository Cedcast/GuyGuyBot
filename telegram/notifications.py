"""
telegram/notifications.py
--------------------------
Pure formatting helpers — no bot state or async dependencies.

All functions return plain strings suitable for Telegram's HTML parse mode.
"""

from __future__ import annotations

from typing import Any


def format_signal_message(signal: dict[str, Any]) -> str:
    """Format a trade signal for Telegram (HTML).

    Parameters
    ----------
    signal:
        Signal dict as stored in the ``signals`` table plus any additional
        keys such as ``signal_id``.

    Returns
    -------
    A multi-line HTML-formatted string.
    """
    direction_emoji = "🟢" if signal.get("direction") == "LONG" else "🔴"
    engine_label = signal.get("engine", "?").capitalize()
    pair = signal.get("pair", "?")
    timeframe = signal.get("timeframe", "?")
    direction = signal.get("direction", "?")
    entry = signal.get("entry", 0.0)
    stop_loss = signal.get("stop_loss", 0.0)
    take_profit = signal.get("take_profit", 0.0)
    confidence = float(signal.get("confidence", 0.0)) * 100
    reasoning = signal.get("reasoning", "—")
    signal_id = signal.get("signal_id") or signal.get("id", "?")

    # Risk/reward ratio (guard against zero denominator)
    try:
        risk = abs(float(entry) - float(stop_loss))
        reward = abs(float(take_profit) - float(entry))
        rr = f"{reward / risk:.1f}" if risk > 0 else "N/A"
    except (TypeError, ZeroDivisionError):
        rr = "N/A"

    return (
        f"{direction_emoji} <b>{pair}</b> — {direction}  "
        f"<i>[{engine_label} · {timeframe}]</i>\n"
        f"\n"
        f"📥 <b>Entry:</b>       <code>{entry}</code>\n"
        f"🛑 <b>Stop Loss:</b>   <code>{stop_loss}</code>\n"
        f"🎯 <b>Take Profit:</b> <code>{take_profit}</code>\n"
        f"⚖️ <b>R:R:</b>         <code>{rr}</code>\n"
        f"🧠 <b>Confidence:</b>  <code>{confidence:.0f}%</code>\n"
        f"\n"
        f"💬 <i>{reasoning}</i>\n"
        f"\n"
        f"<code>Signal #{signal_id}</code>"
    )


def format_trade_outcome_message(trade: dict[str, Any]) -> str:
    """Format a trade close notification for Telegram (HTML).

    Parameters
    ----------
    trade:
        Trade dict as returned by :meth:`~core.signal_logger.SignalLogger.close_trade`.
    """
    outcome = trade.get("outcome", "?")
    pair = trade.get("pair", "?")
    engine = trade.get("engine", "?").capitalize()
    direction = trade.get("direction", "?")
    entry = trade.get("entry", 0.0)
    close_price = trade.get("close_price", 0.0)
    pnl = float(trade.get("pnl") or 0.0)
    r_multiple = float(trade.get("r_multiple") or 0.0)
    trade_id = trade.get("id", "?")

    if outcome == "TP":
        result_emoji = "✅"
        result_text = "Take Profit Hit"
    elif outcome == "SL":
        result_emoji = "❌"
        result_text = "Stop Loss Hit"
    else:
        result_emoji = "ℹ️"
        result_text = outcome

    pnl_sign = "+" if pnl >= 0 else ""

    return (
        f"{result_emoji} <b>{result_text}</b>\n"
        f"\n"
        f"📊 <b>{pair}</b> {direction}  <i>[{engine}]</i>\n"
        f"📥 Entry:       <code>{entry}</code>\n"
        f"📤 Close:       <code>{close_price}</code>\n"
        f"💰 PnL:         <code>{pnl_sign}{pnl:.2f} USD</code>\n"
        f"📐 R-Multiple:  <code>{pnl_sign}{r_multiple:.2f}R</code>\n"
        f"\n"
        f"<code>Trade #{trade_id}</code>"
    )


def format_stats_message(stats: dict[str, Any]) -> str:
    """Format a periodic performance stats message for Telegram (HTML).

    Parameters
    ----------
    stats:
        Stats dict as returned by :meth:`~core.signal_logger.SignalLogger.get_stats`.
    """
    period_type = stats.get("period_type", "?").capitalize()
    period_start = stats.get("period_start", "?")[:10]
    period_end = stats.get("period_end", "?")[:10]
    total_signals = stats.get("total_signals", 0)
    opened = stats.get("opened_trades", 0)
    skipped = stats.get("skipped_trades", 0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    total_pnl = float(stats.get("total_pnl") or 0.0)
    winrate = float(stats.get("winrate") or 0.0) * 100
    avg_r = float(stats.get("avg_r_multiple") or 0.0)
    max_dd = float(stats.get("max_drawdown") or 0.0)

    pnl_sign = "+" if total_pnl >= 0 else ""

    return (
        f"📈 <b>GuyGuyBot — {period_type} Stats</b>\n"
        f"🗓️ {period_start} → {period_end}\n"
        f"\n"
        f"📡 Signals:       <code>{total_signals}</code>\n"
        f"✅ Opened:        <code>{opened}</code>\n"
        f"⏭️ Skipped:       <code>{skipped}</code>\n"
        f"\n"
        f"🏆 Wins:          <code>{wins}</code>\n"
        f"💔 Losses:        <code>{losses}</code>\n"
        f"📊 Win Rate:      <code>{winrate:.1f}%</code>\n"
        f"📐 Avg R:         <code>{avg_r:.2f}R</code>\n"
        f"\n"
        f"💰 Total PnL:     <code>{pnl_sign}{total_pnl:.2f} USD</code>\n"
        f"📉 Max Drawdown:  <code>{max_dd:.2f} USD</code>"
    )


def format_status_message(open_trades: list[dict[str, Any]], version: str = "1.0.0") -> str:
    """Format a /status command reply for Telegram (HTML)."""
    count = len(open_trades)
    lines = [
        f"🤖 <b>GuyGuyBot v{version}</b> — Status\n",
        f"📂 Open trades: <b>{count}</b>\n",
    ]
    if open_trades:
        lines.append("")
        for t in open_trades:
            direction_emoji = "🟢" if t.get("direction") == "LONG" else "🔴"
            lines.append(
                f"{direction_emoji} {t.get('pair')} {t.get('direction')}"
                f" [{t.get('engine')}]  entry={t.get('entry')}"
            )
    else:
        lines.append("No active positions.")
    return "\n".join(lines)
