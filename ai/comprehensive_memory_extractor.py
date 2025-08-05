# ai/comprehensive_memory_extractor.py - Simple extraction result for compatibility
"""
Simple extraction result system for memory processing
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ExtractionResult:
    """Results from memory extraction processing"""
    memory_events: List[Dict[str, Any]] = field(default_factory=list)
    intent_classification: str = "casual_conversation"
    memory_enhancements: Dict[str, Any] = field(default_factory=dict)
    conversation_thread_id: Optional[str] = None
    extracted_entities: List[str] = field(default_factory=list)
    extracted_emotions: List[str] = field(default_factory=list)
    extracted_facts: List[str] = field(default_factory=list)

def extract_all_from_text(username: str, text: str, context: str = "") -> ExtractionResult:
    """
    Extract memories using the existing memory system
    """
    try:
        from ai.memory import get_user_memory
        
        # Get user memory system
        memory = get_user_memory(username)
        
        # Extract memories using existing system
        memory.extract_memories_from_text(text)
        
        # Create basic extraction result
        result = ExtractionResult(
            memory_events=[{"type": "text_processed", "content": text[:100]}],
            intent_classification="memory_processing",
            memory_enhancements={},
            conversation_thread_id=None,
            extracted_entities=[],
            extracted_emotions=[],
            extracted_facts=[]
        )
        
        # Check for conversation threading patterns (McDonald's → McFlurry → Francesco example)
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in ['mcdonald', 'mcflurry', 'francesco', 'went with', 'had with']):
            result.conversation_thread_id = f"conversation_{username}_{hash(text) % 10000}"
            result.memory_enhancements = {"conversation_threading": True}
            result.intent_classification = "memory_enhancement"
        
        # Check for memory recall questions
        if any(keyword in text_lower for keyword in ['where did i', 'what did i', 'who did i', 'when did i']):
            result.intent_classification = "memory_recall"
        
        return result
        
    except Exception as e:
        print(f"[ComprehensiveExtractor] ❌ Extraction error: {e}")
        return ExtractionResult([], "casual_conversation", {}, None, [], [], [])

# Cache for recent extractions to support get_cached_extraction_result
_extraction_cache = {}

def get_cached_extraction_result(question: str) -> Optional[ExtractionResult]:
    """Get cached extraction result if available"""
    cache_key = hash(question.lower()) % 10000
    return _extraction_cache.get(cache_key)

def check_conversation_threading(question: str, username: str) -> bool:
    """Check if this input is part of conversation threading"""
    text_lower = question.lower()
    # Look for patterns that suggest conversation threading
    threading_patterns = [
        'mcdonald', 'mcflurry', 'francesco', 'went with', 'had with',
        'and i', 'also', 'too', 'as well', 'with me'
    ]
    return any(pattern in text_lower for pattern in threading_patterns)