#!/usr/bin/env python3
"""
Buddy Event Logger - JSON event logging for QA analysis
Created for QA system to track all Buddy events in real-time
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import threading
import os

class BuddyEventLogger:
    """
    Simple event logger that writes JSON events to buddy_events.json
    Thread-safe and minimal overhead for QA analysis
    """
    
    def __init__(self, log_file: str = "buddy_events.json"):
        self.log_file = log_file
        self.lock = threading.Lock()
        self.events = []
        self.session_id = f"session_{int(time.time())}"
        
        # Initialize log file
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Initialize the log file with session info"""
        try:
            session_start = {
                "timestamp": datetime.now().isoformat(),
                "event_type": "session_start",
                "session_id": self.session_id,
                "pid": os.getpid(),
                "data": {
                    "start_time": time.time(),
                    "buddy_version": "advanced_ai_consciousness"
                }
            }
            
            with self.lock:
                with open(self.log_file, 'w') as f:
                    json.dump([session_start], f, indent=2)
                    
        except Exception as e:
            print(f"[BuddyEventLogger] ❌ Failed to initialize log file: {e}")
    
    def log_event(self, event_type: str, data: Dict[str, Any], component: str = "unknown"):
        """
        Log an event to the JSON file
        
        Args:
            event_type: Type of event (stt_start, tts_start, etc.)
            data: Event data dictionary
            component: Component that generated the event
        """
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "unix_time": time.time(),
                "event_type": event_type,
                "session_id": self.session_id,
                "component": component,
                "data": data
            }
            
            with self.lock:
                # Read existing events
                try:
                    with open(self.log_file, 'r') as f:
                        events = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    events = []
                
                # Add new event
                events.append(event)
                
                # Keep only last 1000 events to prevent file from growing too large
                if len(events) > 1000:
                    events = events[-1000:]
                
                # Write back to file
                with open(self.log_file, 'w') as f:
                    json.dump(events, f, indent=2)
                    
        except Exception as e:
            print(f"[BuddyEventLogger] ❌ Failed to log event {event_type}: {e}")
    
    def log_stt_start(self, audio_length: int = None):
        """Log STT (Whisper) start event"""
        self.log_event("stt_start", {
            "audio_length": audio_length,
            "start_time": time.time()
        }, "whisper_stt")
    
    def log_stt_finish(self, transcription: str, latency: float, success: bool = True, error: str = None):
        """Log STT (Whisper) finish event"""
        self.log_event("stt_finish", {
            "transcription": transcription,
            "latency": latency,
            "success": success,
            "error": error,
            "transcription_length": len(transcription) if transcription else 0
        }, "whisper_stt")
    
    def log_tts_start(self, text: str, voice: str = None, chunk_id: str = None):
        """Log TTS (Kokoro) synthesis start event"""
        self.log_event("tts_start", {
            "text": text,
            "voice": voice,
            "chunk_id": chunk_id,
            "text_length": len(text),
            "start_time": time.time()
        }, "kokoro_tts")
    
    def log_tts_finish(self, text: str, latency: float, chunk_id: str = None, success: bool = True, error: str = None):
        """Log TTS (Kokoro) synthesis finish event"""
        self.log_event("tts_finish", {
            "text": text,
            "latency": latency,
            "chunk_id": chunk_id,
            "success": success,
            "error": error
        }, "kokoro_tts")
    
    def log_tts_playback_start(self, chunk_id: str = None):
        """Log TTS playback start"""
        self.log_event("tts_playback_start", {
            "chunk_id": chunk_id,
            "start_time": time.time()
        }, "kokoro_tts")
    
    def log_tts_playback_finish(self, chunk_id: str = None, success: bool = True):
        """Log TTS playback finish"""
        self.log_event("tts_playback_finish", {
            "chunk_id": chunk_id,
            "success": success
        }, "kokoro_tts")
    
    def log_tts_queue_event(self, action: str, chunk_id: str = None, queue_size: int = None):
        """Log TTS queue events (queued, played, skipped)"""
        self.log_event("tts_queue", {
            "action": action,  # "queued", "played", "skipped"
            "chunk_id": chunk_id,
            "queue_size": queue_size
        }, "tts_queue")
    
    def log_llm_start(self, prompt: str, user_id: str = None, model: str = None):
        """Log LLM (KoboldCPP) streaming start"""
        self.log_event("llm_start", {
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "prompt_length": len(prompt),
            "user_id": user_id,
            "model": model,
            "start_time": time.time()
        }, "koboldcpp_llm")
    
    def log_llm_finish(self, response: str, token_count: int, tokens_per_sec: float, latency: float, success: bool = True, error: str = None):
        """Log LLM (KoboldCPP) streaming finish"""
        self.log_event("llm_finish", {
            "response": response[:200] + "..." if len(response) > 200 else response,
            "response_length": len(response),
            "token_count": token_count,
            "tokens_per_sec": tokens_per_sec,
            "latency": latency,
            "success": success,
            "error": error
        }, "koboldcpp_llm")
    
    def log_llm_chunk(self, chunk: str, chunk_index: int):
        """Log individual LLM chunk"""
        self.log_event("llm_chunk", {
            "chunk": chunk,
            "chunk_index": chunk_index,
            "chunk_length": len(chunk)
        }, "koboldcpp_llm")
    
    def log_memory_update(self, update_type: str, user_id: str = None, topic: str = None, emotion: str = None, details: Dict[str, Any] = None):
        """Log memory system updates"""
        self.log_event("memory_update", {
            "update_type": update_type,  # "conversation", "user_name", "topic", "emotion"
            "user_id": user_id,
            "topic": topic,
            "emotion": emotion,
            "details": details
        }, "memory_system")
    
    def log_vad_detection(self, action: str, audio_length: int = None, volume: float = None, speech_detected: bool = None):
        """Log VAD (Voice Activity Detection) events"""
        self.log_event("vad_detection", {
            "action": action,  # "start_speaking", "stop_speaking", "processing"
            "audio_length": audio_length,
            "volume": volume,
            "speech_detected": speech_detected
        }, "vad_system")
    
    def log_error(self, error_type: str, error_message: str, component: str, stack_trace: str = None, context: Dict[str, Any] = None):
        """Log runtime errors and warnings"""
        self.log_event("error", {
            "error_type": error_type,  # "runtime_error", "warning", "exception"
            "error_message": error_message,
            "stack_trace": stack_trace,
            "context": context
        }, component)
    
    def log_warning(self, warning_message: str, component: str, context: Dict[str, Any] = None):
        """Log warnings"""
        self.log_event("warning", {
            "warning_message": warning_message,
            "context": context
        }, component)
    
    def log_conversation_start(self, user_id: str = None):
        """Log conversation start"""
        self.log_event("conversation_start", {
            "user_id": user_id
        }, "conversation_manager")
    
    def log_conversation_end(self, user_id: str = None, duration: float = None):
        """Log conversation end"""
        self.log_event("conversation_end", {
            "user_id": user_id,
            "duration": duration
        }, "conversation_manager")
    
    def log_wake_word_detected(self, wake_word: str, confidence: float = None):
        """Log wake word detection"""
        self.log_event("wake_word_detected", {
            "wake_word": wake_word,
            "confidence": confidence
        }, "wake_word_detection")

# Global instance for easy access
buddy_event_logger = BuddyEventLogger()