"""
Unified Memory Manager - Single extraction point for all memory operations
Created: 2025-08-05
Purpose: Eliminate duplicate memory extraction calls across all systems
"""

import time
import hashlib
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict

# Import the existing comprehensive extraction system
from ai.unified_memory_manager import extract_all_from_text, ExtractionResult

@dataclass 
class MemoryExtractionMetadata:
    """Metadata for memory extraction operations"""
    extraction_result: ExtractionResult
    timestamp: float
    username: str
    text: str
    context: str
    extraction_type: str  # "comprehensive", "smart", "fusion"

class UnifiedMemoryManager:
    """
    🎯 SINGLE MEMORY EXTRACTION MANAGER
    Ensures memory extraction happens only once across all systems:
    - main.py 
    - ai/chat_enhanced_smart.py
    - ai/chat_enhanced_smart_with_fusion.py
    """
    
    def __init__(self):
        self.extraction_cache: Dict[str, MemoryExtractionMetadata] = {}
        self.default_cooldown = 10  # 10 seconds default cooldown
        self.max_cache_entries = 20  # Keep memory usage reasonable
        
        print("[UnifiedMemoryManager] 🧠 Initialized single extraction point")
    
    def _generate_cache_key(self, text: str, username: str) -> str:
        """Generate cache key for extraction deduplication"""
        # Create hash from normalized text + username
        normalized_text = text.lower().strip()
        cache_string = f"{username}:{normalized_text}"
        return hashlib.md5(cache_string.encode()).hexdigest()[:16]
    
    def _clean_expired_cache(self, current_time: float, cooldown_seconds: int):
        """Remove expired cache entries"""
        to_remove = []
        for key, metadata in self.extraction_cache.items():
            if current_time - metadata.timestamp > cooldown_seconds:
                to_remove.append(key)
        
        for key in to_remove:
            del self.extraction_cache[key]
            
        # Also limit cache size
        if len(self.extraction_cache) > self.max_cache_entries:
            # Remove oldest entries
            sorted_cache = sorted(
                self.extraction_cache.items(),
                key=lambda x: x[1].timestamp
            )
            entries_to_remove = len(self.extraction_cache) - self.max_cache_entries
            for old_key, _ in sorted_cache[:entries_to_remove]:
                del self.extraction_cache[old_key]
    
    def extract_once(self, text: str, username: str = "Unknown", 
                    cooldown_seconds: int = None, context: str = "", 
                    extraction_type: str = "unified") -> Optional[ExtractionResult]:
        """
        🎯 SINGLE EXTRACTION POINT - prevents duplicate memory extraction
        
        Args:
            text: User input text to extract from
            username: User identifier
            cooldown_seconds: Override default cooldown (10s)
            context: Additional conversation context
            extraction_type: Type of extraction for debugging
        
        Returns:
            ExtractionResult if extraction performed, None if skipped due to cooldown
        """
        if cooldown_seconds is None:
            cooldown_seconds = self.default_cooldown
            
        current_time = time.time()
        cache_key = self._generate_cache_key(text, username)
        
        # Clean expired entries
        self._clean_expired_cache(current_time, cooldown_seconds)
        
        # Check if we have a recent extraction
        if cache_key in self.extraction_cache:
            cached_metadata = self.extraction_cache[cache_key]
            time_since_last = current_time - cached_metadata.timestamp
            
            if time_since_last < cooldown_seconds:
                print(f"[UnifiedMemoryManager] 🔄 SKIPPING duplicate extraction for '{username}': "
                      f"'{text[:30]}...' (last extracted {time_since_last:.1f}s ago)")
                return None  # Return None to indicate skipped extraction
        
        # Perform new extraction using existing unified system
        print(f"[UnifiedMemoryManager] 🧠 EXTRACTING for '{username}' ({extraction_type}): '{text[:50]}...'")
        
        try:
            extraction_result = extract_all_from_text(username, text, context)
            
            # Cache the result
            self.extraction_cache[cache_key] = MemoryExtractionMetadata(
                extraction_result=extraction_result,
                timestamp=current_time,
                username=username,
                text=text,
                context=context,
                extraction_type=extraction_type
            )
            
            print(f"[UnifiedMemoryManager] ✅ Extracted {len(extraction_result.memory_events)} events, "
                  f"intent={extraction_result.intent_classification}")
            
            return extraction_result
            
        except Exception as e:
            print(f"[UnifiedMemoryManager] ❌ Extraction error: {e}")
            # Return empty result to prevent cascade failures
            return ExtractionResult([], "casual_conversation", {}, None, [], [], [])
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics"""
        return {
            "cached_extractions": len(self.extraction_cache),
            "users_in_cache": len(set(meta.username for meta in self.extraction_cache.values())),
            "oldest_cache_age": min([time.time() - meta.timestamp 
                                   for meta in self.extraction_cache.values()]) if self.extraction_cache else 0,
            "extraction_types": list(set(meta.extraction_type for meta in self.extraction_cache.values()))
        }
    
    def clear_user_cache(self, username: str):
        """Clear all cached extractions for a specific user"""
        to_remove = [key for key, meta in self.extraction_cache.items() 
                    if meta.username == username]
        for key in to_remove:
            del self.extraction_cache[key]
        print(f"[UnifiedMemoryManager] 🗑️ Cleared {len(to_remove)} cache entries for {username}")
    
    def force_clear_cache(self):
        """Clear all cached extractions"""
        count = len(self.extraction_cache)
        self.extraction_cache.clear()
        print(f"[UnifiedMemoryManager] 🗑️ Force cleared {count} cache entries")

# Global instance for cross-module access
_global_memory_manager: Optional[UnifiedMemoryManager] = None

def get_memory_manager() -> UnifiedMemoryManager:
    """Get or create global memory manager instance"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = UnifiedMemoryManager()
    return _global_memory_manager

def extract_once(text: str, username: str = "Unknown", cooldown_seconds: int = 10, 
                context: str = "", extraction_type: str = "unified") -> Optional[ExtractionResult]:
    """
    🎯 GLOBAL EXTRACTION FUNCTION - Use this instead of direct memory extraction calls
    
    This function ensures memory extraction happens only once across all systems.
    Replace all calls to:
    - smart_memory.extract_and_store_human_memories()
    - extract_all_from_text()
    - Any other memory extraction functions
    
    With this single function call.
    """
    manager = get_memory_manager()
    return manager.extract_once(text, username, cooldown_seconds, context, extraction_type)

def get_extraction_stats() -> Dict[str, Any]:
    """Get memory extraction statistics"""
    manager = get_memory_manager()
    return manager.get_cache_stats()

def clear_extraction_cache(username: str = None):
    """Clear extraction cache for user or all users"""
    manager = get_memory_manager()
    if username:
        manager.clear_user_cache(username)
    else:
        manager.force_clear_cache()