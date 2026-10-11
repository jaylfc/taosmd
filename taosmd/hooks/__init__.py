"""Hooks for taOS.

Provides infrastructure for parsing Claude Code transcripts and capturing
conversation turns into the archive.
"""

from .parser import parse_entry

__all__ = ["parse_entry"]
