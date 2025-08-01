#!/usr/bin/env python3
"""
Test the buddy event logger to make sure it works
"""

from buddy_event_logger import buddy_event_logger
import time

def test_event_logger():
    print("Testing Buddy Event Logger...")
    
    # Test basic event logging
    buddy_event_logger.log_wake_word_detected("Hey Buddy", 0.95)
    buddy_event_logger.log_conversation_start("test_user")
    
    # Test STT events
    buddy_event_logger.log_stt_start(1600)
    time.sleep(0.1)
    buddy_event_logger.log_stt_finish("Hello, how are you?", 0.5, True)
    
    # Test LLM events
    buddy_event_logger.log_llm_start("Hello, how are you?", "test_user", "consciousness_integrated")
    buddy_event_logger.log_llm_chunk("I'm doing well", 1)
    buddy_event_logger.log_llm_chunk("thank you for asking!", 2)
    buddy_event_logger.log_llm_finish("I'm doing well, thank you for asking!", 15, 2.5, 1.2, True)
    
    # Test TTS events
    buddy_event_logger.log_tts_start("I'm doing well", chunk_id="chunk_1")
    buddy_event_logger.log_tts_finish("I'm doing well", 0.3, "chunk_1", True)
    
    # Test memory update
    buddy_event_logger.log_memory_update("conversation", "test_user", "greeting", "positive")
    
    # Test VAD events
    buddy_event_logger.log_vad_detection("start_speaking", 800, 0.2, True)
    buddy_event_logger.log_vad_detection("stop_speaking", 1600, 0.1, False)
    
    # Test error logging
    buddy_event_logger.log_error("runtime_error", "Test error for QA", "test_component", "Traceback: test")
    buddy_event_logger.log_warning("Test warning", "test_component")
    
    # Test conversation end
    buddy_event_logger.log_conversation_end("test_user", 30.5)
    
    print("✅ Event logging test complete!")
    print("📄 Check buddy_events.json for logged events")

if __name__ == "__main__":
    test_event_logger()