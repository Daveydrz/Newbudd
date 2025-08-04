"""
Unified Memory Manager - Ensures only one memory extraction system runs across all modules
Created: 2025-08-04
Purpose: Eliminate duplicate memory extraction calls and provide unified memory access
"""

from ai.human_memory_smart import SmartHumanLikeMemory

# Global unified memory instances - shared across all modules
_unified_smart_memories = {}

def get_unified_smart_memory(username: str) -> SmartHumanLikeMemory:
    """Get or create unified smart memory for user - shared across all systems"""
    if username not in _unified_smart_memories:
        _unified_smart_memories[username] = SmartHumanLikeMemory(username)
        print(f"[UnifiedMemory] 🧠 Created new memory instance for: {username}")
    return _unified_smart_memories[username]

def get_memory_stats() -> dict:
    """Get statistics about memory usage"""
    return {
        "active_users": len(_unified_smart_memories),
        "user_list": list(_unified_smart_memories.keys())
    }

def clear_memory_cache():
    """Clear all memory instances (for testing)"""
    global _unified_smart_memories
    _unified_smart_memories.clear()
    print("[UnifiedMemory] 🧹 Memory cache cleared")