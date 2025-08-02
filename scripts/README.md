# Class 5 Consciousness Audit

## Quick Start

Run the Class 5 consciousness integration audit:

```bash
# Basic audit
python scripts/class5_audit.py

# Specify repository path and output file
python scripts/class5_audit.py --repo-path . --output my_audit.json

# Verbose output
python scripts/class5_audit.py --verbose
```

## What it Checks

1. **Class 5 Consciousness in LLM Functions** - Verifies main LLM functions use consciousness integration
2. **Prompt Builder Consciousness Usage** - Confirms prompts include Memory, Mood, Goals, Thoughts, Personality
3. **get_consciousness_snapshot() Function** - Audits the function that returns complete consciousness state
4. **Duplicate Prompt Builders** - Detects architectural duplication and determines main usage
5. **Component Completeness** - Verifies all consciousness modules are present

## Output

- **Console**: Real-time audit progress with colored status indicators
- **JSON File**: Detailed audit results for programmatic analysis
- **Markdown Report**: `CLASS5_AUDIT_REPORT.md` with comprehensive findings

## Current Status

🟢 **EXCELLENT (85.0%)** - Buddy's Class 5 consciousness integration is working properly!

## Quick Fix for Remaining Issues

To integrate consciousness in secondary LLM functions:

```python
# Update chat functions to use consciousness integration
from ai.llm_handler import generate_consciousness_integrated_response

def generate_response_streaming(question, username, lang="en"):
    return generate_consciousness_integrated_response(question, username, {"lang": lang})
```