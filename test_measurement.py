#!/usr/bin/env python3
"""Test the measurement script by mocking API calls."""

import datetime
import json
import sys
import os

# Add the script directory to the path
sys.path.insert(0, '/tmp/exec-tsk-rifzf6')

from measure_upstream_propagation import get_reset_boundaries, window_flipped, measure_at_boundary, get_token, fetch_usage


class MockResponse:
    """Mock HTTP response."""
    def __init__(self, returncode, stdout, stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def mock_get_token():
    """Mock token getter."""
    return "test_token"


def mock_fetch_usage(token):
    """Mock fetch usage that simulates window rollover."""
    # Simulate a window flip at a specific time
    current_time = datetime.datetime.now(datetime.timezone.utc)
    
    # Simulate data before and after rollover
    base_data = {
        'five_hour': {
            'utilization': 0.5,
            'resets_at': '2026-08-16T08:00:00+00:00'
        },
        'seven_day': {
            'utilization': 0.3
        }
    }
    
    after_data = {
        'five_hour': {
            'utilization': 0.6,
            'resets_at': '2026-08-16T09:00:00+00:00'
        },
        'seven_day': {
            'utilization': 0.4
        }
    }
    
    # Simulate a rollover after 45 seconds
    if (current_time - datetime.datetime(2026, 8, 16, 7, 59, 59, tzinfo=datetime.timezone.utc)).total_seconds() > 45:
        return after_data
    else:
        return base_data


def test_measurement():
    """Test the measurement with mocked data."""
    print("Testing measurement script with mocked API calls...")
    
    # Patch the functions
    import measure_upstream_propagation
    original_get_token = measure_upstream_propagation.get_token
    original_fetch_usage = measure_upstream_propagation.fetch_usage
    
    measure_upstream_propagation.get_token = mock_get_token
    measure_upstream_propagation.fetch_usage = mock_fetch_usage
    
    try:
        # Test reset boundaries
        boundaries = get_reset_boundaries(datetime.date(2026, 8, 16))
        print(f"\nReset boundaries for 2026-08-16: {[b.isoformat() for _, b in boundaries]}")
        
        # Test measurement at a boundary
        reset_at = datetime.datetime(2026, 8, 16, 8, 0, 0, tzinfo=datetime.timezone.utc)
        
        # Run measurement with a short duration
        delay, samples, details = measure_at_boundary(reset_at, duration=60)
        
        print(f"\nMeasurement results:")
        print(f"  Reset time: {details['reset_at']}")
        print(f"  Flipped at: {details['flipped_at']}")
        print(f"  Delay: {delay} seconds")
        print(f"  Samples: {samples}")
        
        # Determine recommended value
        if delay is None:
            print(f"\nResult: No window flip detected. Recommending keeping MIN_LEAD_SECONDS = 30")
            recommended = 30
        elif delay < 1:
            print(f"\nResult: Delay is effectively zero ({delay:.2f}s). Recommending small non-zero floor")
            recommended = 5
            print(f"  Rationale: Keep a small floor for safety even if propagation is fast")
        elif delay > 30:
            print(f"\nResult: Delay ({delay:.2f}s) exceeds current value (30s)")
            recommended = int(delay) + 5
            print(f"  Rationale: Set above measured delay + safety margin")
        else:
            print(f"\nResult: Delay ({delay:.2f}s) is less than current value (30s)")
            recommended = 30
            print(f"  Rationale: Current value is safe but potentially conservative")
        
        return recommended, details
        
    finally:
        # Restore original functions
        measure_upstream_propagation.get_token = original_get_token
        measure_upstream_propagation.fetch_usage = original_fetch_usage


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING MEASUREMENT SCRIPT")
    print("=" * 60)
    
    try:
        recommended, details = test_measurement()
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Recommended MIN_LEAD_SECONDS: {recommended}")
        print(f"Measurement details: {json.dumps(details, indent=2)}")
        
        # Output the result in a format that can be used
        print(f"\n# Based on measurement:")
        print(f"# Reset: {details['reset_at']}")
        if details['flipped_at']:
            print(f"# Window flipped at: {details['flipped_at']}")
            print(f"# Delay: {details.get('delay_seconds', 'N/A')} seconds")
        
        if recommended != 30:
            print(f"\n# Recommendation: Set MIN_LEAD_SECONDS = {recommended}")
        else:
            print(f"\n# Recommendation: Keep MIN_LEAD_SECONDS = 30 (current value is adequate)")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
