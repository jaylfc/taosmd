"""Tests for the gaming detector."""

import asyncio
import time

from taosmd.gaming_detector import GamingDetector, detect_game_processes


def _run(coro):
    return asyncio.run(coro)


def test_no_games_detected():
    """On a server/Pi, no games should be detected."""
    games = detect_game_processes()
    # Might find something on a dev machine, but shouldn't crash
    assert isinstance(games, list)


def test_detector_idle_state():
    async def run():
        detector = GamingDetector(poll_interval=1, cooldown=0)
        result = await detector.check()
        assert result["status"] == "idle"
        assert not detector.game_active
    _run(run())


def test_manual_yield_and_reclaim():
    async def run():
        detector = GamingDetector(poll_interval=1, cooldown=0)

        result = await detector.force_yield()
        assert result["status"] == "yielded"
        assert detector.game_active

        result = await detector.force_reclaim()
        assert result["status"] == "reclaimed"
        assert not detector.game_active
    _run(run())


def test_cooldown_after_game_exit():
    async def run():
        detector = GamingDetector(poll_interval=1, cooldown=5)

        # Simulate game active then exit
        detector._game_active = True
        detector._game_exited_at = time.time()

        # Should be in cooldown
        result = await detector.check()
        assert result["status"] == "cooldown"
        assert result["seconds_remaining"] > 0

        # Simulate cooldown expired
        detector._game_exited_at = time.time() - 10
        result = await detector.check()
        assert result["status"] == "reclaimed"
        assert not detector.game_active
    _run(run())
