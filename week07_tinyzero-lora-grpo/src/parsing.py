"""Extract Countdown response segments (think block + <answer>)."""

from __future__ import annotations

import re
from dataclasses import dataclass

THINK_OPEN = '<think>'
THINK_CLOSE = '</think>'

_ANSWER_RE = re.compile(r"<answer>(?P<body>.*?)</answer>", re.DOTALL | re.IGNORECASE)

@dataclass
class ParsedResponse:
    has_think_pair: bool
    think_text: str | None
    answer_raw: str | None
    answer_inner: str | None

def parse_countdown_response(text: str, think_open: str = THINK_OPEN, think_close: str = THINK_CLOSE) -> ParsedResponse:
    if text is None:
        text = ""
    t = text.strip()
    # Case: completion starts inside a think block (open tag in prompt)
    has_pair = think_close in t
    think_text: str | None = None
    if has_pair:
        if think_open in t:
            try:
                start = t.index(think_open) + len(think_open)
                end = t.index(think_close, start)
                think_text = t[start:end].strip()
            except ValueError:
                has_pair = False
        else:
            try:
                end = t.index(think_close)
                think_text = t[:end].strip()
            except ValueError:
                has_pair = False

    m = _ANSWER_RE.search(t)
    answer_raw = m.group(0) if m else None
    answer_inner = m.group("body").strip() if m else None
    return ParsedResponse(
        has_think_pair=has_pair,
        think_text=think_text,
        answer_raw=answer_raw,
        answer_inner=answer_inner,
    )

def format_ok_for_reward(
    parsed: ParsedResponse,
    raw_text: str | None = None,
    think_close: str = THINK_CLOSE,
) -> bool:
    if not parsed.has_think_pair or parsed.answer_inner is None:
        return False
    if "=" in parsed.answer_inner:
        return False
    if raw_text is not None:
        try:
            end_think = raw_text.index(think_close) + len(think_close)
        except ValueError:
            return False
        if _ANSWER_RE.search(raw_text[end_think:]) is None:
            return False
    return True