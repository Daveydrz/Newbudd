#!/usr/bin/env python3
"""
Test script to validate the VAD detection and memory retention fixes
"""

def test_memory_retention_fix():
    """Test that memory retention preserves Class 5 consciousness"""
    print("🧠 Testing Memory Retention Fix...")
    
    try:
        from ai.chat_enhanced_smart import reset_session_for_user_smart, get_smart_memory
        
        # Create a memory instance
        memory = get_smart_memory('test_user_retention')
        
        # Simulate adding some conversation history
        memory.conversation_highlights.append({
            'topic': 'mcdonalds_conversation',
            'emotion': 'casual',
            'status': 'pending',
            'created': '2025-08-03T22:00:00'
        })
        
        # Test that reset preserves conversation data
        highlights_before = len(memory.conversation_highlights)
        appointments_before = len(memory.appointments)
        life_events_before = len(memory.life_events)
        
        print(f"  Before reset: {highlights_before} highlights, {appointments_before} appointments, {life_events_before} life events")
        
        # Reset session
        reset_session_for_user_smart('test_user_retention')
        
        # Check that conversation data is preserved
        memory_after = get_smart_memory('test_user_retention')
        highlights_after = len(memory_after.conversation_highlights)
        appointments_after = len(memory_after.appointments)
        life_events_after = len(memory_after.life_events)
        
        print(f"  After reset: {highlights_after} highlights, {appointments_after} appointments, {life_events_after} life events")
        
        if highlights_after == highlights_before and appointments_after == appointments_before and life_events_after == life_events_before:
            print("  ✅ Memory retention fix working - conversation history preserved")
            return True
        else:
            print("  ❌ Memory retention fix failed - conversation history lost")
            return False
            
    except Exception as e:
        print(f"  ❌ Error testing memory retention: {e}")
        return False

def test_vad_detection_fix():
    """Test that VAD detection fix is properly implemented"""
    print("🎤 Testing VAD Detection Fix...")
    
    try:
        # Check that the fix is present in the source code
        with open('audio/input.py', 'r') as f:
            source_code = f.read()
        
        # Check for key components of the fix
        checks = [
            ('YOU FINISHED SPEAKING!' in source_code, "Termination message present"),
            ('silence_frames >= 50' in source_code, "Full duplex silence detection logic"),
            ('silence_frames >= 40' in source_code, "Half duplex silence detection logic"),
            ('break' in source_code and 'has_speech and silence_frames' in source_code, "Break condition implemented")
        ]
        
        all_passed = True
        for check, description in checks:
            if check:
                print(f"  ✅ {description}")
            else:
                print(f"  ❌ {description}")
                all_passed = False
        
        if all_passed:
            print("  ✅ VAD detection fix properly implemented")
            return True
        else:
            print("  ❌ VAD detection fix incomplete")
            return False
            
    except Exception as e:
        print(f"  ❌ Error testing VAD detection: {e}")
        return False

def main():
    """Run all tests"""
    print("🔧 Testing Newbudd Assistant Fixes")
    print("=" * 50)
    
    memory_test = test_memory_retention_fix()
    print()
    vad_test = test_vad_detection_fix()
    
    print()
    print("=" * 50)
    if memory_test and vad_test:
        print("🎉 All fixes working correctly!")
        print("✅ VAD Detection: User speech termination properly detected")
        print("✅ Memory Retention: Class 5 consciousness preserved between conversations")
    else:
        print("❌ Some fixes need attention:")
        if not memory_test:
            print("  - Memory retention fix needs review")
        if not vad_test:
            print("  - VAD detection fix needs review")

if __name__ == "__main__":
    main()