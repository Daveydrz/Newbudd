# ai/streaming_tts_manager.py - Real-time sentence-boundary TTS system
import re
import time
import threading
from typing import List, Optional, Callable
from dataclasses import dataclass
from queue import Queue, Empty

@dataclass
class TTSChunk:
    """TTS chunk for processing"""
    text: str
    is_complete_sentence: bool
    timestamp: float

class StreamingTTSManager:
    """Manages real-time TTS as LLM generates tokens"""
    
    def __init__(self):
        self.tts_queue = Queue()
        self.current_buffer = ""
        self.is_processing = False
        self.tts_thread: Optional[threading.Thread] = None
        self.stop_processing = False
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'([.!?]+(?:\s|$))')
        self.partial_sentence_triggers = re.compile(r'([,:;]\s+)')
        
        # Minimum chunk sizes
        self.min_sentence_length = 10  # Minimum chars for a sentence
        self.max_buffer_size = 200     # Max chars before forcing output
        self.chunk_timeout = 2.0       # Max seconds to wait for complete sentence
        
        self.last_chunk_time = 0
        
    def _get_tts_function(self):
        """Get the TTS function"""
        try:
            from audio.output import speak_streaming
            return speak_streaming
        except ImportError:
            print("[StreamingTTS] ⚠️ TTS function not available")
            return None
    
    def _extract_complete_sentences(self, text: str) -> List[TTSChunk]:
        """Extract complete sentences from text buffer"""
        chunks = []
        
        # Find sentence boundaries
        sentences = self.sentence_endings.split(text)
        
        current_sentence = ""
        for i, part in enumerate(sentences):
            if i % 2 == 0:  # Text part
                current_sentence += part
            else:  # Punctuation part
                current_sentence += part
                # We have a complete sentence
                if len(current_sentence.strip()) >= self.min_sentence_length:
                    chunks.append(TTSChunk(
                        text=current_sentence.strip(),
                        is_complete_sentence=True,
                        timestamp=time.time()
                    ))
                    current_sentence = ""
        
        # Handle remaining text
        remaining_text = current_sentence.strip()
        
        return chunks, remaining_text
    
    def _should_force_output(self, buffer: str) -> bool:
        """Check if we should force output of current buffer"""
        # Force output if buffer is too long
        if len(buffer) > self.max_buffer_size:
            return True
            
        # Force output if we've been waiting too long
        if (time.time() - self.last_chunk_time) > self.chunk_timeout:
            return True
            
        # Force output if we have a meaningful chunk with pause indicators
        if len(buffer) > 30 and self.partial_sentence_triggers.search(buffer):
            return True
            
        return False
    
    def _tts_worker(self):
        """Background worker to process TTS queue"""
        speak_streaming = self._get_tts_function()
        if not speak_streaming:
            print("[StreamingTTS] ❌ TTS worker cannot start - no TTS function")
            return
            
        print("[StreamingTTS] 🎵 TTS worker started")
        
        while not self.stop_processing:
            try:
                # Get next chunk to speak
                chunk = self.tts_queue.get(timeout=0.5)
                
                if chunk.text.strip():
                    print(f"[StreamingTTS] 🗣️ Speaking: '{chunk.text[:50]}...' (complete: {chunk.is_complete_sentence})")
                    
                    # Validate text before speaking
                    if len(chunk.text.strip()) >= 3:  # Minimum 3 chars
                        speak_streaming(chunk.text.strip())
                    else:
                        print(f"[StreamingTTS] ⚠️ Skipping short chunk: '{chunk.text}'")
                
                self.tts_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                print(f"[StreamingTTS] ❌ TTS worker error: {e}")
                
        print("[StreamingTTS] 🛑 TTS worker stopped")
    
    def start_session(self):
        """Start TTS session"""
        if self.is_processing:
            return
            
        print("[StreamingTTS] ▶️ Starting TTS session")
        self.is_processing = True
        self.stop_processing = False
        self.current_buffer = ""
        self.last_chunk_time = time.time()
        
        # Start background TTS worker
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
    
    def add_token(self, token: str):
        """Add token from LLM stream"""
        if not self.is_processing:
            return
            
        self.current_buffer += token
        self.last_chunk_time = time.time()
        
        # Extract complete sentences
        complete_chunks, remaining_buffer = self._extract_complete_sentences(self.current_buffer)
        
        # Queue complete sentences for TTS
        for chunk in complete_chunks:
            self.tts_queue.put(chunk)
        
        # Update buffer with remaining text
        self.current_buffer = remaining_buffer
        
        # Check if we should force output of remaining buffer
        if self.current_buffer and self._should_force_output(self.current_buffer):
            print(f"[StreamingTTS] ⏰ Forcing output: '{self.current_buffer[:30]}...'")
            force_chunk = TTSChunk(
                text=self.current_buffer,
                is_complete_sentence=False,
                timestamp=time.time()
            )
            self.tts_queue.put(force_chunk)
            self.current_buffer = ""
    
    def add_text_chunk(self, text: str):
        """Add larger text chunk (multiple tokens)"""
        if not text:
            return
            
        # Process token by token to maintain real-time processing
        for char in text:
            self.add_token(char)
    
    def finish_session(self):
        """Finish TTS session and speak any remaining text"""
        if not self.is_processing:
            return
            
        print("[StreamingTTS] 🏁 Finishing TTS session")
        
        # Speak any remaining buffer content
        if self.current_buffer.strip():
            final_chunk = TTSChunk(
                text=self.current_buffer.strip(),
                is_complete_sentence=False,
                timestamp=time.time()
            )
            self.tts_queue.put(final_chunk)
            self.current_buffer = ""
        
        # Wait for queue to empty
        try:
            self.tts_queue.join()
        except:
            pass
            
        # Stop processing
        self.stop_processing = True
        self.is_processing = False
        
        # Wait for worker thread to finish
        if self.tts_thread and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=2.0)
        
        print("[StreamingTTS] ✅ TTS session finished")
    
    def stop_session(self):
        """Stop TTS session immediately"""
        print("[StreamingTTS] 🛑 Stopping TTS session")
        self.stop_processing = True
        self.is_processing = False
        
        # Clear queue
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except Empty:
                break
    
    def get_stats(self) -> dict:
        """Get TTS processing statistics"""
        return {
            'is_processing': self.is_processing,
            'buffer_length': len(self.current_buffer),
            'queue_size': self.tts_queue.qsize(),
            'worker_alive': self.tts_thread.is_alive() if self.tts_thread else False
        }

# Global streaming TTS manager
streaming_tts_manager = StreamingTTSManager()

def start_streaming_tts_session():
    """Start streaming TTS session"""
    streaming_tts_manager.start_session()

def add_llm_token_to_tts(token: str):
    """Add LLM token to TTS stream"""
    streaming_tts_manager.add_token(token)

def add_llm_chunk_to_tts(chunk: str):
    """Add LLM chunk to TTS stream"""
    streaming_tts_manager.add_text_chunk(chunk)

def finish_streaming_tts_session():
    """Finish streaming TTS session"""
    streaming_tts_manager.finish_session()

def stop_streaming_tts_session():
    """Stop streaming TTS session immediately"""
    streaming_tts_manager.stop_session()

def get_tts_stats() -> dict:
    """Get TTS statistics"""
    return streaming_tts_manager.get_stats()