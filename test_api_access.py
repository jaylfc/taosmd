#!/usr/bin/env python3
"""Quick measurement test with real API."""
import subprocess
import json
import sys

def test_api_call():
    """Test if we can reach the API with the proper credential handling."""
    
    # Read the token from credentials
    with open('/home/jay/.claude/.credentials.json', 'r') as f:
        creds = json.load(f)
    
    token = creds['claudeAiOauth']['accessToken']
    
    # Use the exact same format as usage_publish.sh lines 20-21
    cmd = (
        f'printf "header = \"Authorization: Bearer {token}\"\\n'
        f'header = \"anthropic-beta: oauth-2025-04-20\"\\n"\\n'
        f'| curl -s -m 15 --config - https://api.anthropic.com/api/oauth/usage'
    )
    
    print("Testing API call...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and result.stdout:
            print(f"API call successful! Response preview:")
            print(result.stdout[:200])
            
            # Parse the response
            data = json.loads(result.stdout)
            if 'five_hour' in data:
                print(f"\nFive-hour window:")
                print(f"  Utilization: {data['five_hour'].get('utilization', 'N/A')}")
                print(f"  Resets_at: {data['five_hour'].get('resets_at', 'N/A')}")
            
            return True, data
        else:
            print(f"API call failed (return code: {result.returncode})")
            print(f"Error: {result.stderr}")
            return False, None
            
    except Exception as e:
        print(f"Error during API call: {e}")
        return False, None

if __name__ == "__main__":
    success, data = test_api_call()
    if success:
        print("\n✅ API access confirmed!")
        sys.exit(0)
    else:
        print("\n❌ API access failed!")
        sys.exit(1)
