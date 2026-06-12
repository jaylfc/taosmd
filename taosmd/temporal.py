"""Natural language temporal expression parsing for retrieval-time date filtering.

This module is the retrieval-time complement to temporal_boost.py.  Where
temporal_boost reranks by keyword signals, this module parses *explicit* date
range expressions from the query and applies them as a precision filter or
score boost.

The lever is default-off: pass ``temporal={"window": "...", "mode": "filter"}``
(or ``"mode": "boost"``) to ``retrieve()`` to activate it.  Full-scale
validation on LoCoMo-1540 is still pending.

Functions
---------
parse_temporal_expression(expression, now) -> (from_dt, to_dt) | None
extract_temporal_expression(query) -> str | None
parse_hit_datetime(value) -> datetime | None
apply_temporal_stage(hits, temporal, query, logger) -> list[dict]
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging

# ---------------------------------------------------------------------------
# Month name lookup
# ---------------------------------------------------------------------------

_MONTHS: dict[str, int] = {
    "january": 0, "february": 1, "march": 2, "april": 3,
    "may": 4, "june": 5, "july": 6, "august": 7,
    "september": 8, "october": 9, "november": 10, "december": 11,
    "jan": 0, "feb": 1, "mar": 2, "apr": 3,
    # "may" intentionally absent as 3-letter alias: too many false positives
    # as a modal verb.  Only matched via full name or explicit date patterns.
    "jun": 5, "jul": 6, "aug": 7, "sep": 8, "sept": 8,
    "oct": 9, "nov": 10, "dec": 11,
}

_MONTH_PATTERN = "|".join(sorted(_MONTHS.keys(), key=len, reverse=True))

# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------


def _day_range(d: datetime) -> tuple[datetime, datetime]:
    """Full-day range 00:00:00.000000 through 23:59:59.999999."""
    start = d.replace(hour=0, minute=0, second=0, microsecond=0)
    end = d.replace(hour=23, minute=59, second=59, microsecond=999999)
    return start, end


def _week_range(d: datetime) -> tuple[datetime, datetime]:
    """Monday-through-Sunday week range containing d.

    weekday() returns 0 for Monday, 6 for Sunday, so Monday offset = -weekday().
    """
    monday = d - timedelta(days=d.weekday())
    start = monday.replace(hour=0, minute=0, second=0, microsecond=0)
    end = (monday + timedelta(days=6)).replace(
        hour=23, minute=59, second=59, microsecond=999999
    )
    return start, end


def _month_range(year: int, month_0: int) -> tuple[datetime, datetime]:
    """Full range for a calendar month (month_0 is 0-based January)."""
    month_1 = month_0 + 1
    start = datetime(year, month_1, 1, 0, 0, 0, 0)
    # Last day: first day of next month minus one day.
    if month_1 == 12:
        last_day = datetime(year, 12, 31)
    else:
        last_day = datetime(year, month_1 + 1, 1) - timedelta(days=1)
    end = last_day.replace(hour=23, minute=59, second=59, microsecond=999999)
    return start, end


def _year_range(year: int) -> tuple[datetime, datetime]:
    """Full calendar year range."""
    start = datetime(year, 1, 1, 0, 0, 0, 0)
    end = datetime(year, 12, 31, 23, 59, 59, 999999)
    return start, end


def _quarter_range(year: int, quarter: int) -> tuple[datetime, datetime]:
    """Q1-Q4 range (1-based quarter)."""
    start_month_0 = (quarter - 1) * 3  # 0-based
    # Quarter spans 3 months: start_month_0 through start_month_0+2.
    end_month_0 = start_month_0 + 2
    start = datetime(year, start_month_0 + 1, 1, 0, 0, 0, 0)
    return_end = _month_range(year, end_month_0)[1]
    return start, return_end


def _ago_range(ref: datetime, amount: int, unit: str) -> tuple[datetime, datetime]:
    """Range from <amount> <unit>s before ref (floored to day-start) through ref."""
    if unit == "day":
        from_dt = ref - timedelta(days=amount)
    elif unit == "week":
        from_dt = ref - timedelta(weeks=amount)
    elif unit == "month":
        # Same calendar day in the past month; clamp to month-end if needed.
        year = ref.year
        month = ref.month - amount
        while month <= 0:
            month += 12
            year -= 1
        import calendar
        max_day = calendar.monthrange(year, month)[1]
        day = min(ref.day, max_day)
        from_dt = ref.replace(year=year, month=month, day=day)
    elif unit == "hour":
        from_dt = ref - timedelta(hours=amount)
    else:
        from_dt = ref - timedelta(days=amount)

    from_dt = from_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return from_dt, ref


# ---------------------------------------------------------------------------
# Public: parse_temporal_expression
# ---------------------------------------------------------------------------


def parse_temporal_expression(
    expression: str,
    now: datetime | None = None,
) -> tuple[datetime, datetime] | None:
    """Parse a natural-language temporal expression into a (from, to) range.

    Args:
        expression: A temporal expression such as "last week", "in May 2023",
            "3 days ago", "Q1 2026", "8 May 2023", "before last month".
        now: Reference datetime (naive).  Defaults to datetime.now().
            Always pass an explicit value in tests for determinism.

    Returns:
        A (from_dt, to_dt) tuple of naive datetimes, or None if the
        expression cannot be parsed.
    """
    ref = now if now is not None else datetime.now()
    inp = expression.strip().lower()

    # Simple keywords
    if inp == "today":
        return _day_range(ref)

    if inp == "yesterday":
        return _day_range(ref - timedelta(days=1))

    if inp in ("this week", "last week"):
        d = ref if inp == "this week" else ref - timedelta(days=7)
        return _week_range(d)

    if inp == "this month":
        return _month_range(ref.year, ref.month - 1)

    if inp == "last month":
        d = ref
        month_0 = d.month - 2  # subtract 1 for 0-based, 1 more for "last"
        year = d.year
        while month_0 < 0:
            month_0 += 12
            year -= 1
        return _month_range(year, month_0)

    if inp == "this year":
        return _year_range(ref.year)

    if inp == "last year":
        return _year_range(ref.year - 1)

    # "N days/weeks/months/hours ago"
    m = re.fullmatch(
        r"(\d+)\s+(day|days|week|weeks|month|months|hour|hours)\s+ago", inp
    )
    if m:
        amount = int(m.group(1))
        unit = m.group(2).rstrip("s")
        return _ago_range(ref, amount, unit)

    # "last N days/weeks/months"
    m = re.fullmatch(r"last\s+(\d+)\s+(day|days|week|weeks|month|months)", inp)
    if m:
        amount = int(m.group(1))
        unit = m.group(2).rstrip("s")
        return _ago_range(ref, amount, unit)

    # "in <month>" or "in <month>[,] [year]"
    m = re.fullmatch(r"in\s+(" + _MONTH_PATTERN + r")(?:[,]?\s+(\d{4}))?", inp)
    if m:
        month_0 = _MONTHS[m.group(1)]
        year = int(m.group(2)) if m.group(2) else ref.year
        return _month_range(year, month_0)

    # Explicit day dates: "8 May 2023", "8 May, 2023", "May 8 2023", "May 8, 2023",
    # "on 8 May, 2023"
    # Pattern: optional "on ", then either (day month [,] year) or (month day[,] year)
    m = re.fullmatch(
        r"(?:on\s+)?(\d{1,2})\s+(" + _MONTH_PATTERN + r")[,]?\s+(\d{4})", inp
    )
    if m:
        day = int(m.group(1))
        month_0 = _MONTHS[m.group(2)]
        year = int(m.group(3))
        try:
            d = datetime(year, month_0 + 1, day)
        except ValueError:
            pass
        else:
            return _day_range(d)

    m = re.fullmatch(
        r"(?:on\s+)?(" + _MONTH_PATTERN + r")\s+(\d{1,2})[,]?\s+(\d{4})", inp
    )
    if m:
        month_0 = _MONTHS[m.group(1)]
        day = int(m.group(2))
        year = int(m.group(3))
        try:
            d = datetime(year, month_0 + 1, day)
        except ValueError:
            pass
        else:
            return _day_range(d)

    # Bare month name (must not be "may" without explicit context)
    if inp in _MONTHS and inp != "may":
        return _month_range(ref.year, _MONTHS[inp])

    # "Q1 2026", "q2 2025"
    m = re.fullmatch(r"q([1-4])\s+(\d{4})", inp)
    if m:
        return _quarter_range(int(m.group(2)), int(m.group(1)))

    # "before <expr>" — from datetime.min to inner.from
    m = re.fullmatch(r"before\s+(.+)", inp)
    if m:
        inner = parse_temporal_expression(m.group(1), ref)
        if inner:
            return datetime.min, inner[0]

    # "after <expr>" — from inner.to to ref
    m = re.fullmatch(r"after\s+(.+)", inp)
    if m:
        inner = parse_temporal_expression(m.group(1), ref)
        if inner:
            return inner[1], ref

    return None


# ---------------------------------------------------------------------------
# Public: extract_temporal_expression
# ---------------------------------------------------------------------------

# Compiled regex: ordered longest/most-specific first.
# Bare month names are only matched with an "in " prefix or inside explicit
# dates to avoid false positives ("may" as modal verb, "May" as a name, etc.).
_EXTRACT_RE = re.compile(
    r"""
    (?:
        # before/after + inner expressions (recursive not possible here; match
        # common single-token sub-expressions after the keyword)
        \b(?:before|after)\s+
        (?:
            last\s+\d+\s+(?:day|days|week|weeks|month|months)
            | \d+\s+(?:day|days|week|weeks|month|months|hour|hours)\s+ago
            | (?:last|this)\s+(?:week|month|year)
            | today|yesterday
            | in\s+(?:""" + _MONTH_PATTERN + r""")(?:[,]?\s+\d{4})?
            | (?:""" + _MONTH_PATTERN + r""")(?:[,]?\s+\d{4})?
            | \d{1,2}\s+(?:""" + _MONTH_PATTERN + r""")[,]?\s+\d{4}
            | (?:""" + _MONTH_PATTERN + r""")\s+\d{1,2}[,]?\s+\d{4}
            | Q[1-4]\s+\d{4}
        )
        |
        # "last N days/weeks/months" (before "last week/month/year" so longer wins)
        \blast\s+\d+\s+(?:day|days|week|weeks|month|months)\b
        |
        # "N units ago"
        \b\d+\s+(?:day|days|week|weeks|month|months|hour|hours)\s+ago\b
        |
        # "last/this week/month/year"
        \b(?:last|this)\s+(?:week|month|year)\b
        |
        # today / yesterday
        \b(?:today|yesterday)\b
        |
        # Quarters
        \bQ[1-4]\s+\d{4}\b
        |
        # "in <month> [year]" (with optional comma before year)
        \bin\s+(?:""" + _MONTH_PATTERN + r""")(?:[,]?\s+\d{4})?\b
        |
        # Explicit day dates with optional "on ": "8 May 2023", "May 8, 2023",
        # "on 20 July, 2023"
        \b(?:on\s+)?\d{1,2}\s+(?:""" + _MONTH_PATTERN + r""")[,]?\s+\d{4}\b
        |
        \b(?:on\s+)?(?:""" + _MONTH_PATTERN + r""")\s+\d{1,2}[,]?\s+\d{4}\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def extract_temporal_expression(query: str) -> str | None:
    """Scan a natural-language query for the first embedded temporal expression.

    Args:
        query: A full query string such as "what did we discuss last week?".

    Returns:
        The matched substring (suitable for passing to parse_temporal_expression),
        or None if no supported expression is found.

    Note:
        Bare month names are intentionally NOT matched as free-standing words
        because "May" is too common as a name or modal verb.  They are matched
        only inside "in <month>" or explicit date patterns.
    """
    m = _EXTRACT_RE.search(query)
    return m.group(0).strip() if m else None


# ---------------------------------------------------------------------------
# Public: parse_hit_datetime
# ---------------------------------------------------------------------------

_LOCOMO_FORMATS = [
    "%I:%M %p on %d %B, %Y",   # "8:56 pm on 20 July, 2023"
    "%d %B, %Y",                # "20 July, 2023"
    "%d %B %Y",                 # "20 July 2023"
]


def parse_hit_datetime(value: object) -> datetime | None:
    """Parse a stored memory timestamp into a naive datetime.

    Accepts:
    - int or float: treated as POSIX epoch seconds.
    - str that parses as float: same.
    - ISO-8601 strings (with or without trailing "Z" or timezone offset).
    - LoCoMo conversation format: "8:56 pm on 20 July, 2023".

    Returns:
        A naive datetime, or None if the value cannot be parsed.
        Never raises.
    """
    if value is None:
        return None

    try:
        # Numeric epoch
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value))

        if not isinstance(value, str):
            return None

        s = value.strip()
        if not s:
            return None

        # String that looks like a float (epoch)
        try:
            return datetime.fromtimestamp(float(s))
        except (ValueError, OSError, OverflowError):
            pass

        # ISO-8601: replace trailing Z, then strip tzinfo to keep naive
        iso = s
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(iso)
            return dt.replace(tzinfo=None)
        except ValueError:
            pass

        # LoCoMo conversation format: normalise am/pm to uppercase for strptime
        # (strptime %p is locale-dependent; normalise to upper for safety)
        s_norm = re.sub(r"\b(am|pm)\b", lambda mo: mo.group(0).upper(), s, flags=re.IGNORECASE)
        for fmt in _LOCOMO_FORMATS:
            try:
                return datetime.strptime(s_norm, fmt)
            except ValueError:
                continue

    except Exception:  # noqa: BLE001
        pass

    return None


# ---------------------------------------------------------------------------
# Public: apply_temporal_stage
# ---------------------------------------------------------------------------


def apply_temporal_stage(
    hits: list[dict],
    temporal: dict,
    query: str,
    logger: "logging.Logger | None" = None,
) -> list[dict]:
    """Apply a temporal filter or boost stage to retrieved hits.

    This is the retrieval-pipeline integration point.  It is designed to be
    injected just before the final ``[:limit]`` truncation inside retrieve().

    Args:
        hits: Normalised result dicts, each with at minimum a "score" key
            and optional "metadata" / "created_at" fields.
        temporal: Configuration dict.  Recognised keys:

            * ``window`` (str, optional): explicit temporal expression, e.g.
              "in May 2023" or "last week".  Takes precedence over auto-scan.
            * ``auto`` (bool, default False): when True and no window given,
              run extract_temporal_expression(query) to find an expression.
            * ``mode`` (str, "boost" or "filter", default "boost").
            * ``boost`` (float, default 0.25): in boost mode, multiply in-range
              hit scores by ``(1 + boost)``.
            * ``reference``: reference "now" for relative expressions.  Accepts
              epoch float, ISO string, LoCoMo-format string, or datetime.
              Defaults to the real clock.
            * ``datetime_keys`` (list[str], default ``["datetime", "created_at"]``):
              metadata keys to check in order for each hit's timestamp.

        query: Original query string (used only when ``auto=True``).
        logger: Optional logger for debug messages.

    Returns:
        The hits list, potentially filtered or reordered.  Fail-open: any
        unexpected error returns the original list unchanged.

    Note:
        Default is off.  Full-scale validation on LoCoMo-1540 is pending.
        Enabling this without benchmarking on your data is discouraged.
    """
    try:
        return _apply_temporal_stage_inner(hits, temporal, query, logger)
    except Exception:  # noqa: BLE001
        return hits


def _apply_temporal_stage_inner(
    hits: list[dict],
    temporal: dict,
    query: str,
    logger: "logging.Logger | None",
) -> list[dict]:
    # 1. Resolve expression string.
    expression: str | None = temporal.get("window")
    if not expression and temporal.get("auto", False):
        expression = extract_temporal_expression(query)

    if not expression:
        if logger:
            logger.debug("temporal: no expression found, stage skipped")
        return hits

    # 2. Resolve reference time.
    reference_raw = temporal.get("reference")
    if reference_raw is None:
        ref_now = datetime.now()
    elif isinstance(reference_raw, datetime):
        ref_now = reference_raw
    else:
        ref_now = parse_hit_datetime(reference_raw) or datetime.now()

    # 3. Parse expression to range.
    parsed = parse_temporal_expression(expression, ref_now)
    if parsed is None:
        if logger:
            logger.debug("temporal: expression %r unparseable, stage skipped", expression)
        return hits

    from_dt, to_dt = parsed

    # 4. Config.
    mode = temporal.get("mode", "boost")
    boost_factor = float(temporal.get("boost", 0.25))
    datetime_keys: list[str] = temporal.get("datetime_keys", ["datetime", "created_at"])

    def _resolve_hit_dt(hit: dict) -> datetime | None:
        meta = hit.get("metadata", {}) or {}
        # After _adapt_vector normalisation, hit["metadata"] is the whole raw
        # vector row dict, which nests the original user metadata under its own
        # "metadata" key.  Build a prioritised lookup chain: nested user-metadata
        # first (has LoCoMo datetime strings), then the outer meta dict, then the
        # hit's top-level created_at.
        nested_meta = meta.get("metadata", {}) or {}
        for candidate in (nested_meta, meta):
            for key in datetime_keys:
                raw = candidate.get(key)
                if raw is not None:
                    dt = parse_hit_datetime(raw)
                    if dt is not None:
                        return dt
        # Top-level hit created_at (used by some source adapters)
        raw = hit.get("created_at")
        if raw is not None:
            return parse_hit_datetime(raw)
        return None

    if mode == "filter":
        filtered = []
        for hit in hits:
            dt = _resolve_hit_dt(hit)
            if dt is None:
                # Fail-open: keep hits with no parseable timestamp.
                filtered.append(hit)
            elif from_dt <= dt <= to_dt:
                filtered.append(hit)

        # Fail-open globally: if filtering would leave zero hits, return original.
        if not filtered:
            if logger:
                logger.debug(
                    "temporal: filter would leave 0 hits, returning original list"
                )
            return hits
        return filtered

    # Boost mode (default).
    modified = list(hits)
    for hit in modified:
        dt = _resolve_hit_dt(hit)
        if dt is None:
            continue
        if from_dt <= dt <= to_dt:
            current_score = hit.get("score", 0.0)
            if "score" in hit:
                hit["score"] = current_score * (1.0 + boost_factor)

    # Re-sort descending by score.
    modified.sort(key=lambda h: h.get("score", 0.0), reverse=True)
    return modified
