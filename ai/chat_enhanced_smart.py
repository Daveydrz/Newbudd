# ai/chat_enhanced_smart.py - Smart LLM-based chat integration
import random
from ai.chat import generate_response_streaming, ask_kobold_streaming, get_current_brisbane_time
from ai.human_memory_smart import SmartHumanLikeMemory
from ai.memory import add_to_conversation_history
from ai.unified_memory_manager import get_unified_smart_memory

def get_smart_memory(username: str) -> SmartHumanLikeMemory:
    """Get or create smart memory for user - uses unified memory manager"""
    return get_unified_smart_memory(username)

def reset_session_for_user_smart(username: str):
    """Reset session when conversation starts"""
    memory = get_smart_memory(username)
    memory.reset_session_context()

def generate_response_streaming_with_smart_memory(question, username, lang="en", context=None):
    """Streaming version with smart LLM-based memory + 8k context window management"""
    try:
        print(f"[SmartChat] 🧠 Starting smart LLM-based streaming for '{question}' from {username}")
        
        # Check context window limits before processing (8k token management)
        try:
            from ai.context_window_manager import check_context_window_rollover, create_context_snapshot_for_user
            
            # Build current context from previous conversation if available
            current_context = context.get("current_context", "") if context else ""
            if not current_context:
                # Build minimal context for new conversations
                current_context = f"System: You are Buddy, an AI assistant.\nUser: {question}"
            
            needs_rollover, fresh_context = check_context_window_rollover(username, current_context, question)
            
            if needs_rollover:
                print(f"[SmartChat] 🔄 Context window rollover triggered for {username}")
                
                # Create snapshot to preserve memory context
                conversation_history = context.get("conversation_history", []) if context else []
                working_memory = context.get("working_memory", {}) if context else {}
                
                snapshot_created = create_context_snapshot_for_user(
                    username, current_context, working_memory, conversation_history
                )
                
                if snapshot_created:
                    print(f"[SmartChat] 📸 Context snapshot created - memories preserved across 8k limit")
                    # Update context to use fresh compressed context with memory injection
                    if context:
                        context["current_context"] = fresh_context
                        context["context_rollover_occurred"] = True
                    else:
                        context = {"current_context": fresh_context, "context_rollover_occurred": True}
                else:
                    print(f"[SmartChat] ⚠️ Context snapshot failed - using standard compression")
            
        except ImportError:
            print(f"[SmartChat] ⚠️ Context window manager not available - using standard processing")
        
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