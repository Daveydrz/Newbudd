#!/usr/bin/env python3
"""
Test script to verify unified memory extraction prevents duplicates
Simulates the three locations mentioned by the user calling memory extraction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai.memory_manager import extract_once, get_extraction_stats, clear_extraction_cache
import time

def simulate_main_py_extraction(text, username):
    """Simulate main.py calling memory extraction"""
    print(f"[MAIN.PY] 🧠 Safety memory extraction for '{username}': '{text[:30]}...'")
    result = extract_once(text, username, cooldown_seconds=10, extraction_type="main_safety")
    if result is not None:
        print(f"[MAIN.PY] ✅ Extracted {len(result.memory_events)} events")
        return True  # Extraction performed
    else:
        print(f"[MAIN.PY] 🔄 Skipped (already extracted)")
        return False  # Extraction skipped

def simulate_chat_enhanced_smart_extraction(text, username):
    """Simulate chat_enhanced_smart.py calling memory extraction"""
    print(f"[SMART] 🧠 Smart memory extraction for '{username}': '{text[:30]}...'")
    result = extract_once(text, username, cooldown_seconds=10, extraction_type="smart_memory")
    if result is not None:
        print(f"[SMART] ✅ Extracted {len(result.memory_events)} events")
        return True  # Extraction performed
    else:
        print(f"[SMART] 🔄 Skipped (already extracted)")
        return False  # Extraction skipped

def simulate_chat_enhanced_smart_with_fusion_extraction(text, username):
    """Simulate chat_enhanced_smart_with_fusion.py calling memory extraction"""
    print(f"[FUSION] 🧠 Fusion memory extraction for '{username}': '{text[:30]}...'")
    result = extract_once(text, username, cooldown_seconds=10, extraction_type="fusion")
    if result is not None:
        print(f"[FUSION] ✅ Extracted {len(result.memory_events)} events")
        return True  # Extraction performed
    else:
        print(f"[FUSION] 🔄 Skipped (already extracted)")
        return False  # Extraction skipped

def test_unified_memory_extraction():
    """Test that multiple systems don't create duplicate extractions"""
    print("🧪 TESTING UNIFIED MEMORY EXTRACTION")
    print("=" * 50)
    
    # Clear any existing cache
    clear_extraction_cache()
    
    test_text = "I went to McDonald's earlier today with Francesco"
    test_username = "TestUser"
    
    print(f"\n📝 Test scenario: User says '{test_text}'")
    print("🎯 Expected: Only ONE extraction should happen, others should be skipped\n")
    
    # Simulate the three systems calling extraction (as mentioned in the issue)
    print("1️⃣ Main.py calls memory extraction:")
    main_result = simulate_main_py_extraction(test_text, test_username)
    
    print("\n2️⃣ Chat Enhanced Smart calls memory extraction:")
    smart_result = simulate_chat_enhanced_smart_extraction(test_text, test_username)
    
    print("\n3️⃣ Chat Enhanced Smart with Fusion calls memory extraction:")
    fusion_result = simulate_chat_enhanced_smart_with_fusion_extraction(test_text, test_username)
    
    # Show statistics
    print(f"\n📊 RESULTS:")
    stats = get_extraction_stats()
    print(f"   Cached extractions: {stats['cached_extractions']}")
    print(f"   Users in cache: {stats['users_in_cache']}")
    print(f"   Extraction types recorded: {stats['extraction_types']}")
    
    # Verify only one extraction happened
    extractions_performed = sum([1 for result in [main_result, smart_result, fusion_result] if result])
    extractions_skipped = sum([1 for result in [main_result, smart_result, fusion_result] if not result])
    
    print(f"\n✅ VERIFICATION:")
    print(f"   Extractions performed: {extractions_performed}")
    print(f"   Extractions skipped: {extractions_skipped}")
    
    if extractions_performed == 1 and extractions_skipped == 2:
        print("🎉 SUCCESS: Only one extraction performed, duplicates properly prevented!")
    else:
        print("❌ FAILURE: Multiple extractions occurred!")
    
    print("\n" + "=" * 50)
    
    # Test with different user to verify it still works
    print("\n🧪 TESTING WITH DIFFERENT USER")
    different_user = "OtherUser"
    print(f"📝 Test scenario: Different user '{different_user}' says same text")
    print("🎯 Expected: New extraction should happen\n")
    
    print("4️⃣ Different user - Main.py calls memory extraction:")
    different_user_result = simulate_main_py_extraction(test_text, different_user)
    
    if different_user_result:
        print("✅ SUCCESS: Different user triggered new extraction!")
    else:
        print("❌ FAILURE: Different user extraction was incorrectly skipped!")

if __name__ == "__main__":
    test_unified_memory_extraction()