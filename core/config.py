"""
core/config.py
--------------
Loads and validates config.yaml, exposing a typed Config dataclass.
Call `load_config()` once at startup; pass the returned object everywhere.

Environment variable overrides
-------------------------------
API keys can be provided via environment variables instead of (or in
addition to) config.yaml.  Environment variables take precedence:

  TELEGRAM_BOT_TOKEN   → telegram.bot_token
  TELEGRAM_CHAT_ID     → telegram.chat_id
  CLAUDE_API_KEY       → llm.claude_api_key
  GPT4O_API_KEY        → llm.gpt4o_api_key
  GROK_API_KEY         → llm.grok_api_key
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


# ---------------------------------------------------------------------------
# Typed sub-config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TelegramStatsSchedule:
    weekly: bool = True
    monthly: bool = True
    quarterly: bool = True


@dataclass
class TelegramConfig:
    bot_token: str
    chat_id: str
    stats_schedule: TelegramStatsSchedule = field(default_factory=TelegramStatsSchedule)


@dataclass
class DatabaseConfig:
    path: str = "data/guybot.db"


@dataclass
class TradingConfig:
    pairs: list[str] = field(default_factory=list)
    risk_per_trade: float = 0.01
    account_size: float = 1000.0


@dataclass
class EngineSettings:
    enabled: bool = True
    timeframes: list[str] = field(default_factory=list)
    scan_interval: int = 60


@dataclass
class EnginesConfig:
    scalping: EngineSettings = field(default_factory=EngineSettings)
    swing: EngineSettings = field(default_factory=EngineSettings)


@dataclass
class LLMConfig:
    primary_agents: list[str] = field(default_factory=lambda: ["claude", "gpt4o"])
    arbitration_agent: str = "grok"
    claude_api_key: str = ""
    gpt4o_api_key: str = ""
    grok_api_key: str = ""


@dataclass
class NewsConfig:
    enabled: bool = True
    cache_ttl_seconds: int = 300
    cryptopanic_token: str = "Pub"
    sentiment_warning_threshold: float = -0.5
    fear_greed_warning_threshold: int = 20


@dataclass
class PaperTradingConfig:
    enabled: bool = True
    start_date: str | None = None
    note: str = "30-day paper trading period. No real orders executed."


@dataclass
class Config:
    """Top-level config object. Constructed by :func:`load_config`."""

    telegram: TelegramConfig
    database: DatabaseConfig
    trading: TradingConfig
    engines: EnginesConfig
    llm: LLMConfig
    news: NewsConfig = field(default_factory=NewsConfig)
    paper_trading: PaperTradingConfig = field(default_factory=PaperTradingConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _parse_telegram(raw: dict[str, Any]) -> TelegramConfig:
    schedule_raw = raw.get("stats_schedule", {})
    schedule = TelegramStatsSchedule(
        weekly=schedule_raw.get("weekly", True),
        monthly=schedule_raw.get("monthly", True),
        quarterly=schedule_raw.get("quarterly", True),
    )
    # Environment variables override config.yaml values
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN") or raw["bot_token"]
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or str(raw["chat_id"])
    return TelegramConfig(
        bot_token=bot_token,
        chat_id=chat_id,
        stats_schedule=schedule,
    )


def _parse_engines(raw: dict[str, Any]) -> EnginesConfig:
    def _es(section: dict[str, Any], default_tf: list[str], default_interval: int) -> EngineSettings:
        return EngineSettings(
            enabled=section.get("enabled", True),
            timeframes=section.get("timeframes", default_tf),
            scan_interval=section.get("scan_interval", default_interval),
        )

    return EnginesConfig(
        scalping=_es(raw.get("scalping", {}), ["1m", "5m", "15m", "30m"], 60),
        swing=_es(raw.get("swing", {}), ["4h", "1d"], 3600),
    )


def _parse_news(raw: dict[str, Any]) -> NewsConfig:
    return NewsConfig(
        enabled=raw.get("enabled", True),
        cache_ttl_seconds=int(raw.get("cache_ttl_seconds", 300)),
        cryptopanic_token=raw.get("cryptopanic_token", "Pub"),
        sentiment_warning_threshold=float(raw.get("sentiment_warning_threshold", -0.5)),
        fear_greed_warning_threshold=int(raw.get("fear_greed_warning_threshold", 20)),
    )


def _parse_paper_trading(raw: dict[str, Any]) -> PaperTradingConfig:
    return PaperTradingConfig(
        enabled=raw.get("enabled", True),
        start_date=raw.get("start_date"),
        note=raw.get("note", "30-day paper trading period. No real orders executed."),
    )


def load_config(path: Path | str = CONFIG_PATH) -> Config:
    """Load and parse ``config.yaml`` from *path*.

    Returns a fully-typed :class:`Config` object.  Raises ``FileNotFoundError``
    if the config file is missing and ``ValueError`` for structural errors.

    Environment variables override the values in ``config.yaml``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    logger.info("Loaded config from %s", path)

    try:
        telegram = _parse_telegram(raw["telegram"])
    except KeyError as exc:
        raise ValueError(f"Missing required config key: {exc}") from exc

    db_raw = raw.get("database", {})
    trading_raw = raw.get("trading", {})
    llm_raw = raw.get("llm", {})
    news_raw = raw.get("news", {})
    pt_raw = raw.get("paper_trading", {})

    # Resolve LLM API keys: env vars take precedence over config.yaml
    claude_key = os.environ.get("CLAUDE_API_KEY") or llm_raw.get("claude_api_key", "")
    gpt4o_key = os.environ.get("GPT4O_API_KEY") or llm_raw.get("gpt4o_api_key", "")
    grok_key = os.environ.get("GROK_API_KEY") or llm_raw.get("grok_api_key", "")

    return Config(
        telegram=telegram,
        database=DatabaseConfig(path=db_raw.get("path", "data/guybot.db")),
        trading=TradingConfig(
            pairs=trading_raw.get("pairs", []),
            risk_per_trade=float(trading_raw.get("risk_per_trade", 0.01)),
            account_size=float(trading_raw.get("account_size", 1000.0)),
        ),
        engines=_parse_engines(raw.get("engines", {})),
        llm=LLMConfig(
            primary_agents=llm_raw.get("primary_agents", ["claude", "gpt4o"]),
            arbitration_agent=llm_raw.get("arbitration_agent", "grok"),
            claude_api_key=claude_key,
            gpt4o_api_key=gpt4o_key,
            grok_api_key=grok_key,
        ),
        news=_parse_news(news_raw),
        paper_trading=_parse_paper_trading(pt_raw),
    )
