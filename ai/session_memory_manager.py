# ai/session_memory_manager.py - Single-source memory extraction prevention system
import time
from typing import Dict, Optional, Any
from dataclasses import dataclass
from ai.unified_memory_manager import extract_all_from_text, ExtractionResult

@dataclass
class SessionExtractionCache:
    """Cache for session-based memory extraction"""
    extraction_result: ExtractionResult
    timestamp: float
    username: str
    question: str
    context: str

class SessionMemoryManager:
    """Ensures memory extraction happens only once per conversation turn"""
    
    def __init__(self):
        self.extraction_cache: Dict[str, SessionExtractionCache] = {}
        self.session_timeout = 45  # 45 seconds timeout for same session
        
    def _generate_cache_key(self, username: str, question: str, context: str = "") -> str:
        """Generate cache key for extraction"""
        # Use first 100 chars of question + username + short context hash
        question_short = question[:100]
        context_hash = str(hash(context[:200])) if context else "no_context"
        return f"{username}_{hash(question_short)}_{context_hash}"
    
    def _is_cache_valid(self, cache_entry: SessionExtractionCache) -> bool:
        """Check if cache entry is still valid"""
        return (time.time() - cache_entry.timestamp) < self.session_timeout
    
    def get_or_extract_memory(self, username: str, question: str, context: str = "") -> ExtractionResult:
        """Get cached extraction or perform new extraction if needed"""
        cache_key = self._generate_cache_key(username, question, context)
        
        # Check if we have a valid cached result
        if cache_key in self.extraction_cache:
            cache_entry = self.extraction_cache[cache_key]
            if self._is_cache_valid(cache_entry):
                print(f"[SessionMemoryManager] ♻️ Using cached extraction for {username}")
                return cache_entry.extraction_result
            else:
                # Remove expired cache
                del self.extraction_cache[cache_key]
                print(f"[SessionMemoryManager] 🗑️ Removed expired cache for {username}")
        
        # Perform new extraction
        print(f"[SessionMemoryManager] 🧠 Performing new memory extraction for {username}")
        extraction_result = extract_all_from_text(username, question, context)
        
        # Cache the result
        self.extraction_cache[cache_key] = SessionExtractionCache(
            extraction_result=extraction_result,
            timestamp=time.time(),
            username=username,
            question=question,
            context=context
        )
        
        # Clean up old cache entries (keep only last 10)
        if len(self.extraction_cache) > 10:
            # Remove oldest entries
            sorted_cache = sorted(
                self.extraction_cache.items(),
                key=lambda x: x[1].timestamp
            )
            for old_key, _ in sorted_cache[:-10]:
                del self.extraction_cache[old_key]
        
        return extraction_result
    
    def clear_cache_for_user(self, username: str):
        """Clear all cache entries for a specific user"""
        to_remove = [key for key, cache in self.extraction_cache.items() 
                    if cache.username == username]
        for key in to_remove:
            del self.extraction_cache[key]
        print(f"[SessionMemoryManager] 🧹 Cleared cache for {username}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        valid_entries = sum(1 for cache in self.extraction_cache.values() 
                          if self._is_cache_valid(cache))
        return {
            'total_entries': len(self.extraction_cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self.extraction_cache) - valid_entries
        }

# Global session manager instance
session_memory_manager = SessionMemoryManager()

def get_session_memory_extraction(username: str, question: str, context: str = "") -> ExtractionResult:
    """Main function to get memory extraction (prevents duplicates)"""
    return session_memory_manager.get_or_extract_memory(username, question, context)

def reset_session_memory_for_user(username: str):
    """Reset memory session for a user (call when conversation ends)"""
    session_memory_manager.clear_cache_for_user(username)