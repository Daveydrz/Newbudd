# ai/chat_enhanced_smart.py - Smart LLM-based chat integration
import random
from ai.chat import generate_response_streaming, ask_kobold_streaming, get_current_brisbane_time
from ai.human_memory_smart import SmartHumanLikeMemory
from ai.memory import add_to_conversation_history

# Global smart memory instances
smart_memories = {}

def get_smart_memory(username: str) -> SmartHumanLikeMemory:
    """Get or create smart memory for user"""
    if username not in smart_memories:
        smart_memories[username] = SmartHumanLikeMemory(username)
    return smart_memories[username]

def reset_session_for_user_smart(username: str):
    """Reset session when conversation starts - PRESERVES CONVERSATION HISTORY FOR CLASS 5 CONSCIOUSNESS"""
    memory = get_smart_memory(username)
    # ✅ FIXED: Only reset session context tracking, NOT conversation history
    # This preserves Class 5 consciousness and memory continuity between turns
    # The context_used_this_session is for avoiding repeating the same memory references
    # within a single session, but conversation history should persist
    memory.reset_session_context()
    
    # ✅ ENHANCED: Ensure the mega memory system also maintains continuity
    # The mega_memory should not be reset between conversation turns
    if hasattr(memory, 'mega_memory'):
        # The mega memory system maintains its own conversation context
        # We don't reset it to preserve Class 5 consciousness
        print(f"[SmartChat] 🧠 Class 5 consciousness maintained for {username}")
    
    print(f"[SmartChat] 🔄 Session reset for {username} with preserved conversation history")

def generate_response_streaming_with_smart_memory(question, username, lang="en"):
    """Streaming version with smart LLM-based memory"""
    try:
        print(f"[SmartChat] 🧠 Starting smart LLM-based streaming for '{question}' from {username}")
        
        # Get smart memory
        smart_memory = get_smart_memory(username)
        
        # Smart LLM-based memory extraction
        smart_memory.extract_and_store_human_memories(question)
        
        # Check for natural context response
        context_response = smart_memory.check_for_natural_context_response()
        
        # If we have natural context, yield it first
        if context_response:
            print(f"[SmartChat] 💭 Smart memory response: {context_response}")
            yield context_response
            
            import time
            time.sleep(0.3)
            
            connectors = [" ", "Also, ", "And ", "By the way, ", "Oh, and "]
            yield random.choice(connectors)
        
        # Use existing streaming generation
        full_response = ""
        for chunk in generate_response_streaming(question, username, lang):
            if chunk and chunk.strip():
                full_response += chunk.strip() + " "
                yield chunk.strip()
        
        # Add to conversation history
        if full_response.strip():
            complete_response = context_response + " " + full_response if context_response else full_response
            add_to_conversation_history(username, question, complete_response.strip())
        
    except Exception as e:
        print(f"[SmartChat] ❌ Error: {e}")
        yield "Sorry, I'm having trouble thinking right now."