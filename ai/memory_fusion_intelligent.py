# ai/memory_fusion_intelligent.py - Simple username unification
"""
Simple memory fusion system for username unification
"""

def get_intelligent_unified_username(username: str, skip_fusion: bool = False) -> str:
    """
    Get unified username from memory fusion system
    
    Args:
        username: Original username
        skip_fusion: Whether to skip fusion processing
        
    Returns:
        Unified username (same as input for now)
    """
    if skip_fusion:
        # Skip fusion processing when requested
        return username
    
    # For now, return the same username
    # In a full implementation, this would check for user consolidation/fusion
    return username