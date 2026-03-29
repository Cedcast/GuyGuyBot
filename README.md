# GuyGuyBot 🤖📈

A modular, multi-agent, multi-LLM trading signal system for Binance Futures/Margin — featuring a Scalping engine, a Swing engine, three LLM agents (Claude, GPT-5, Grok), Telegram integration with inline trade buttons, SQLite persistence, and a risk gate that enforces one open paper trade per pair/engine.

---

## Features

| Feature | Detail |
|---|---|
| **Dual engines** | Scalping (1m–30m) and Swing (4h–1d) |
| **Multi-LLM pipeline** | Claude + GPT-5 run in parallel; Grok arbitrates on disagreement |
| **Telegram signals** | Entry / SL / TP with ✅ Opened / ❌ Skipped inline buttons |
| **Paper trade tracking** | One open trade per pair/engine; TP & SL close notifications |
| **Signal journal** | Win-rate, R-multiple, drawdown stats (weekly/monthly/quarterly) |
| **SQLite persistence** | Survives restarts; WAL mode for reliability |
| **Extensible agents** | Drop in any new LLM agent by subclassing `BaseAgent` |
| **Central config** | Single `config.yaml` for all settings |

---

## Project Structure

```
GuyGuyBot/
├── README.md
├── requirements.txt
├── config.yaml                # ← Central config (edit this first)
├── main.py                    # Entry point
├── core/
│   ├── config.py              # Typed config loader
│   ├── database.py            # SQLite schema + connection manager
│   └── signal_logger.py       # Signal / trade / stats CRUD
├── engines/
│   ├── base_engine.py         # Abstract base engine
│   ├── scalping_engine.py     # 1m–30m scanner (stub market data)
│   └── swing_engine.py        # 4h–1d scanner (stub market data)
├── agents/
│   ├── base_agent.py          # Abstract base agent
│   ├── claude_agent.py        # Anthropic Claude stub
│   ├── gpt_agent.py           # OpenAI GPT-5 stub
│   ├── grok_agent.py          # xAI Grok stub
│   └── pipeline.py            # Orchestrator (parallel + arbitration)
├── telegram/
│   ├── bot.py                 # Bot, handlers, scheduler jobs
│   └── notifications.py       # HTML message formatters
├── risk/
│   └── gating.py              # One-trade-per-pair/engine gate
└── data/
    └── guybot.db              # Auto-created SQLite database
```

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-org/GuyGuyBot.git
cd GuyGuyBot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure `config.yaml`

Open `config.yaml` and replace every `YOUR_*` placeholder:

```yaml
telegram:
  bot_token: "123456:ABC-your-bot-token"
  chat_id:   "987654321"

llm:
  claude_api_key: "sk-ant-..."
  gpt5_api_key:   "sk-..."
  grok_api_key:   "xai-..."
```

All other settings (pairs, risk, scan intervals) are already sensible defaults.

### 3. Run

```bash
python main.py
```

GuyGuyBot will:
1. Create `data/guybot.db` automatically.
2. Connect to Telegram.
3. Start scanning pairs with the stub engines.
4. Send signal messages (with buttons) to your configured chat.

---

## Configuration Reference

| Key | Default | Description |
|---|---|---|
| `telegram.bot_token` | — | BotFather token |
| `telegram.chat_id` | — | Chat or group ID to send messages to |
| `telegram.stats_schedule.weekly` | `true` | Send weekly stats on Monday 09:00 UTC |
| `telegram.stats_schedule.monthly` | `true` | Send monthly stats on the 1st at 09:00 UTC |
| `telegram.stats_schedule.quarterly` | `true` | Send quarterly stats on quarter start |
| `database.path` | `data/guybot.db` | Path to the SQLite file |
| `trading.pairs` | 7 major pairs | Symbols to scan |
| `trading.risk_per_trade` | `0.01` | Risk fraction per trade (1 %) |
| `trading.account_size` | `1000.0` | Paper account USD size |
| `engines.scalping.enabled` | `true` | Toggle scalping engine |
| `engines.scalping.scan_interval` | `60` | Seconds between scans |
| `engines.swing.enabled` | `true` | Toggle swing engine |
| `engines.swing.scan_interval` | `3600` | Seconds between scans |
| `llm.primary_agents` | `[claude, gpt5]` | Two agents run per signal |
| `llm.arbitration_agent` | `grok` | Agent that breaks ties |

---

## Plugging In Your Own Agent Logic

Each LLM agent is a simple class in `agents/`. Open e.g. `agents/claude_agent.py`:

```python
async def analyze(self, market_data, context):
    # TODO: Replace with real Anthropic API call, e.g.:
    #
    # client = anthropic.Anthropic(api_key=self.api_key)
    # message = client.messages.create(
    #     model="claude-opus-4-5",
    #     max_tokens=1024,
    #     system=_SYSTEM_PROMPT,
    #     messages=[{"role": "user", "content": str(market_data)}],
    # )
    # return json.loads(message.content[0].text)
```

Replace the `# TODO` block with your real API call. The method must return:

```python
{
    "direction":   "LONG" | "SHORT" | "NEUTRAL",
    "confidence":  float,   # 0.0–1.0
    "entry":       float,
    "stop_loss":   float,
    "take_profit": float,
    "reasoning":   str,
}
```

### Adding a new LLM agent

1. Create `agents/my_agent.py` and subclass `BaseAgent`.
2. Implement `name`, `analyze`, and `summarize`.
3. Add the agent key to `_agent_classes` in `agents/pipeline.py`.
4. Reference it in `config.yaml` under `llm.primary_agents` or `llm.arbitration_agent`.

---

## Plugging In Real Market Data

Engine scan methods call `_fetch_market_data` which is stubbed. To use real Binance data:

```python
# engines/scalping_engine.py  (and swing_engine.py)
from binance import AsyncClient

async def _fetch_market_data(self, pair: str, timeframe: str):
    client = await AsyncClient.create(api_key, api_secret)
    klines = await client.get_klines(symbol=pair, interval=timeframe, limit=100)
    # Parse klines -> { "pair", "timeframe", "close", "open", "high", "low", "volume", "indicators" }
    ...
```

---

## How the Signal Pipeline Works

```
Engine.scan()
   └─ For each candidate:
        ├─ RiskGate.can_trade(pair, engine)?  →  No → skip
        └─ AgentPipeline.run_pipeline(market_data)
              ├─ Primary Agent 1  ─┐
              ├─ Primary Agent 2  ─┤  (run concurrently)
              └─ Agree?           ─┘
                    ├─ Yes → merge & threshold check → Signal
                    └─ No  → Arbitration Agent → Signal | None

Signal logged to DB → Telegram message with buttons
User taps [✅ Opened Trade] → trade record created + RiskGate locked
User taps [❌ Skipped Trade] → signal marked SKIPPED
Trade monitor polls prices → TP/SL hit → close trade → notify Telegram → RiskGate released
```

---

## Database Schema

| Table | Purpose |
|---|---|
| `signals` | Every generated signal with status lifecycle |
| `trades` | Open/closed paper trades with PnL and R-multiple |
| `stats` | Computed weekly/monthly/quarterly performance snapshots |
| `positions` | Active pair/engine locks (RiskGate persistence) |

---

## Telegram Commands

| Command | Description |
|---|---|
| `/start` | Welcome message |
| `/status` | List currently open paper trades |
| `/stats` | Show the latest weekly performance stats |

---

## Requirements

- Python 3.10+
- See `requirements.txt` for Python packages
- A Telegram bot token (from [@BotFather](https://t.me/BotFather))
- API keys for Claude, GPT-5, and/or Grok (only those you intend to use)

---

## License

MIT
