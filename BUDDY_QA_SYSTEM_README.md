# Buddy QA System

This QA system monitors Buddy's operation in real-time and detects issues like double-speaking, transcription failures, TTS overlaps, LLM stalls, and memory update problems.

## Files Created

1. **`buddy_event_logger.py`** - JSON event logging system integrated into main.py
2. **`buddy_qa_agent.py`** - QA agent that launches Buddy and analyzes logs
3. **`test_event_logger.py`** - Test script for the event logger
4. **`test_qa_agent.py`** - Test script for the QA agent

## How It Works

### Step 1: Event Logging in main.py

The system adds minimal logging hooks to main.py without changing core logic:

- **Whisper STT**: Logs start/finish with transcription text and latency
- **Kokoro TTS**: Logs synthesis and playback events with chunk IDs and latency  
- **KoboldCPP LLM**: Logs streaming start/finish with token count and tokens/sec
- **Memory Updates**: Logs what changed (user name, topics, emotions)
- **VAD Detection**: Logs when user starts/stops speaking with audio length
- **TTS Queue**: Logs when chunks are queued, played, or skipped
- **Runtime Errors**: Logs any warnings or exceptions with stack traces

All events are written to `buddy_events.json` with timestamps.

### Step 2: QA Agent Monitoring

The QA agent (`buddy_qa_agent.py`):

- Launches main.py as a subprocess
- Captures stdout, stderr, and reads buddy_events.json in real-time
- Analyzes events for patterns and issues
- Generates comprehensive reports

### Step 3: Issue Detection

The analyzer automatically detects:

- **Double-speaking**: Multiple TTS chunks playing simultaneously
- **Whisper failures**: STT transcription errors or timeouts
- **Kokoro overlaps**: TTS playback overlapping issues
- **KoboldCPP stalls**: LLM responses taking too long
- **Memory failures**: Memory system update errors
- **Infinite loops**: Consciousness loops blocking wake word detection
- **Response timeouts**: Conversations starting but no LLM response
- **TTS queue issues**: Audio chunks being skipped

## Usage

### Quick Test (No Real Buddy Launch)

```bash
# Test the event logger
python test_event_logger.py

# Test the QA agent analysis
python test_qa_agent.py
```

### Monitor Buddy for 10 Minutes

```bash
python buddy_qa_agent.py --duration 10
```

### Monitor Buddy Continuously

```bash
python buddy_qa_agent.py --continuous
```

### Monitor Custom Script

```bash
python buddy_qa_agent.py --buddy-script custom_buddy.py --duration 5
```

## Output Files

After running the QA agent:

- **`buddy_events.json`** - Real-time event log from Buddy
- **`buddy_qa_report.txt`** - Human-readable analysis report
- **`session_logs.json`** - Detailed JSON session data

## Example Report

```
============================================================
BUDDY QA ANALYSIS REPORT
============================================================
Generated: 2025-08-01T09:27:06
Session Duration: 120.5 seconds
Total Events Logged: 247

ISSUES DETECTED:
------------------------------
Total Issues: 3
  ❌ Double Speaking: 1
  ❌ Runtime Errors: 2

LATENCY BREAKDOWN:
------------------------------
Average STT (Whisper) Latency: 0.850s
Average TTS (Kokoro) Latency: 0.320s
Average LLM (KoboldCPP) Latency: 2.100s

OPTIMIZATION SUGGESTIONS:
------------------------------
  • Fix double speaking: Add proper TTS queue management
  • Improve STT reliability: Check microphone input quality
```

## Integration with main.py

The logging hooks are added strategically without changing core logic:

```python
# QA logging for STT events
if QA_LOGGING_ENABLED:
    buddy_event_logger.log_stt_start(audio_length)
    # ... processing ...
    buddy_event_logger.log_stt_finish(transcription, latency, success)

# QA logging for LLM events  
if QA_LOGGING_ENABLED:
    buddy_event_logger.log_llm_start(text, user_id, model)
    # ... LLM processing ...
    buddy_event_logger.log_llm_finish(response, token_count, tokens_per_sec, latency)

# QA logging for memory updates
if QA_LOGGING_ENABLED:
    buddy_event_logger.log_memory_update("conversation", user_id, topic, emotion)
```

## Benefits

1. **Non-invasive**: Minimal impact on Buddy's performance
2. **Comprehensive**: Tracks all major subsystems
3. **Real-time**: Detects issues as they happen
4. **Actionable**: Provides specific optimization suggestions
5. **Automated**: No manual intervention required

## Detected Issues Examples

- **"Double speaking detected: 12 chunks in 0.1s"** - TTS queue overflow
- **"Whisper STT failure: Connection timeout"** - Audio input problems  
- **"TTS overlap detected: 5 starts vs 2 finishes"** - Audio playback sync issues
- **"LLM stall: 15.2s latency"** - KoboldCPP performance problems
- **"Potential infinite loop: Describe this experience..."** - Consciousness loop blocking wake word

The QA system ensures Buddy operates reliably and helps identify performance bottlenecks before they impact user experience.