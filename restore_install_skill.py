# First, let's see what's in the current cli.py around _install_skill_cmd
import sys
sys.path.insert(0, '.')

with open('taosmd/cli.py', 'r') as f:
    content = f.read()

# Find the _install_skill_cmd function
start = content.find('def _install_skill_cmd')
if start == -1:
    print("Function not found")
    sys.exit(1)

# Extract until the next function
end = content.find('\n\n\n', start)
if end == -1:
    end = len(content)

function_text = content[start:end]
print("Current _install_skill_cmd function:")
print("=" * 80)
print(function_text)
print("=" * 80)

# Now let's see what the manifest handling logic should be
print("\n\nWe need to check if manifest handling logic exists elsewhere or restore it from 78b3dd0e")
print("Let's check git history for the function...")

# Check if there are any references to manifest in the file
if 'manifest' in content.lower():
    print("Found 'manifest' references in the file")
else:
    print("No 'manifest' references found in the file")

# Check if .taosmd-skill-manifest.json is referenced
if '.taosmd-skill-manifest.json' in content:
    print("Found .taosmd-skill-manifest.json reference")
else:
    print("No .taosmd-skill-manifest.json reference found")
