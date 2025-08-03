# ai/chat.py - Enhanced LLM chat integration with Memory + Smart Location & Time + ULTRA-RESPONSIVE STREAMING
import re
import requests
import json
from datetime import datetime
import pytz
from ai.memory import get_conversation_context, get_user_memory
from config import *
from typing import Dict, Any

def _generate_dynamic_error_response(error_context: Dict[str, Any]) -> str:
    """Generate dynamic, personalized error responses using LLM instead of hardcoded messages"""
    try:
        # Try to import consciousness and LLM modules
        try:
            from ai.conscious_prompt_builder import ConsciousPromptBuilder
            from ai.llm_handler import LLMHandler
            
            # Create consciousness-aware error response
            builder = ConsciousPromptBuilder()
            
            error_prompts = {
                'connection_error': "I'm having trouble connecting to my processing systems right now. Express this naturally without being technical.",
                'timeout_error': "My response is taking longer than expected. Acknowledge this in a natural, personal way.",
                'json_decode_error': "I received information I couldn't process properly. Express this conversationally.",
                'no_choices': "My processing didn't generate the expected response format. Communicate this naturally.",
                'http_error': "There's a technical issue with my response generation. Express this in a friendly way.",
                'streaming_error': "Something went wrong while I was generating my response. Acknowledge this naturally.",
                'response_generation_error': "I encountered an issue while thinking through my response. Express this conversationally.",
                'general_error': "I ran into an unexpected issue. Express this in a natural, personal way.",
                'unexpected_error': "Something unexpected happened on my end. Communicate this naturally."
            }
            
            error_type = error_context.get('error_type', 'general_error')
            error_prompt = error_prompts.get(error_type, error_prompts['general_error'])
            
            # Build consciousness-aware prompt
            consciousness_prompt = f"""You are Buddy, an AI assistant experiencing a technical issue. 

Context: {error_context}

Instruction: {error_prompt}

Respond as yourself with your natural personality - be authentic, not overly apologetic, and maintain your conversational style. Keep it brief and natural."""
            
            # Try to get LLM response
            try:
                llm_handler = LLMHandler()
                response_generator = llm_handler.generate_response_with_consciousness(
                    consciousness_prompt, "system", {"context": "error_handling", "use_optimization": False}, 
                    stream=False, is_primary_call=False
                )
                # Collect all response chunks
                response_chunks = []
                for chunk in response_generator:
                    if chunk:
                        response_chunks.append(chunk)
                response = "".join(response_chunks).strip()
                if response:
                    return response
            except Exception as e:
                print(f"[Chat] ⚠️ LLM error handling failed: {e}")
                pass
                
        except ImportError:
            pass
        
        # Fallback to simple dynamic responses (but still more natural than hardcoded)
        error_type = error_context.get('error_type', 'general_error')
        
        fallback_responses = {
            'connection_error': "Having some connection issues on my end - give me a moment.",
            'timeout_error': "This is taking longer than usual - let me try again.",
            'json_decode_error': "Got some garbled info back - let me process that differently.",
            'no_choices': "My response didn't come through right - trying again.",
            'http_error': "Hit a technical snag - working on it.",
            'streaming_error': "Something hiccupped while I was responding.",
            'response_generation_error': "My thinking got a bit tangled there.",
            'general_error': "Something went sideways on my end.",
            'unexpected_error': "That wasn't supposed to happen - let me sort this out."
        }
        
        return fallback_responses.get(error_type, "Something's not quite right - give me a sec.")
        
    except Exception as e:
        print(f"[ErrorResponse] ❌ Error generating dynamic error response: {e}")
        return "Give me a moment to sort this out."

# Import time and location helpers
try:
    from utils.time_helper import get_time_info_for_buddy, get_buddy_current_time, get_buddy_location
    LOCATION_HELPERS_AVAILABLE = True
except ImportError:
    LOCATION_HELPERS_AVAILABLE = False
    print("[Chat] ⚠️ Location helpers not available, using fallback")

def get_current_brisbane_time():
    """Get current Brisbane time - UPDATED to 6:59 PM Brisbane"""
    try:
        brisbane_tz = pytz.timezone('Australia/Brisbane')
        # Current UTC time: 08:59:59 = 6:59 PM Brisbane
        current_time = datetime.now(brisbane_tz)
        return {
            'datetime': current_time.strftime("%Y-%m-%d %H:%M:%S"),
            'time_12h': current_time.strftime("%I:%M %p"),
            'time_24h': current_time.strftime("%H:%M"),
            'date': current_time.strftime("%A, %B %d, %Y"),
            'day': current_time.strftime("%A"),
            'timezone': 'Australia/Brisbane (+10:00)'
        }
    except:
        # Fallback with current time
        return {
            'datetime': "2025-07-06 18:59:59",
            'time_12h': "6:59 PM",
            'time_24h': "18:59",
            'date': "Sunday, July 6, 2025",
            'day': "Sunday",
            'timezone': 'Australia/Brisbane (+10:00)'
        }

def generate_response_streaming(question, username, lang=DEFAULT_LANG):
    """✅ CONSCIOUSNESS-INTEGRATED: Generate AI response with consciousness integration and streaming"""
    try:
        print(f"[ChatStream] 🧠 Starting consciousness-integrated streaming for '{question}' from user '{username}'")
        
        # Import LLMHandler for consciousness integration
        from ai.llm_handler import LLMHandler
        
        # 🔧 FIX: Check for unified username from memory fusion
        try:
            from ai.memory_fusion_intelligent import get_intelligent_unified_username
            unified_username = get_intelligent_unified_username(username)
            if unified_username != username:
                print(f"[ChatStream] 🎯 Using unified username: {username} → {unified_username}")
                username = unified_username
        except ImportError:
            print(f"[ChatStream] ⚠️ Memory fusion not available, using original username: {username}")
        
        # 🎯 NEW: Smart name handling - avoid Anonymous_001
        display_name = None
        use_name = False
        
        try:
            from voice.database import anonymous_clusters, known_users
            
            # Check if this is a named cluster
            if username.startswith('Anonymous_'):
                cluster_data = anonymous_clusters.get(username, {})
                assigned_name = cluster_data.get('test_name', '')
                if assigned_name and assigned_name != 'Unknown':
                    display_name = assigned_name
                    use_name = True
                    print(f"[ChatStream] 👤 Using assigned name: {display_name}")
                else:
                    print(f"[ChatStream] 🚫 Avoiding anonymous cluster name: {username}")
                    use_name = False
            elif username in known_users:
                display_name = username
                use_name = True
                print(f"[ChatStream] 👤 Using known user name: {display_name}")
            else:
                print(f"[ChatStream] 👤 No specific name handling for: {username}")
                display_name = username
                use_name = True
        
        except Exception as e:
            print(f"[ChatStream] ⚠️ Name resolution error: {e}")
            display_name = username if not username.startswith('Anonymous_') else None
            use_name = display_name is not None
        
        # Get current time info (only when needed)
        try:
            from utils.location_manager import get_time_info, get_precise_location_summary
            time_info = get_time_info()
            current_location = get_precise_location_summary()
        except Exception as e:
            print(f"[ChatStream] ⚠️ Location helper failed: {e}")
            brisbane_time = get_current_brisbane_time()
            time_info = brisbane_time
            current_location = "Brisbane, Queensland, Australia"
        
        # Build conversation context
        print(f"[ChatStream] 📚 Getting conversation context...")
        context = get_conversation_context(username)
        
        # Get user memory for additional context
        print(f"[ChatStream] 🧠 Getting user memory...")
        memory = get_user_memory(username)
        reminders = memory.get_today_reminders()
        follow_ups = memory.get_follow_up_questions()
        
        # 🧠 WORKING MEMORY: Get natural language context for LLM
        natural_context = memory.get_natural_language_context_for_llm(question)
        print(f"[ChatStream] 🔗 Working memory context: {natural_context[:100]}..." if natural_context else "[ChatStream] 🔗 No working memory context")
        
        # Build reminder text (optimized)
        reminder_text = ""
        if reminders:
            top_reminders = reminders[:2]
            reminder_text = f"\nImportant stuff for today: {', '.join(top_reminders)}"
        
        # Build follow-up text (optimized)
        follow_up_text = ""
        if follow_ups:
            follow_up_text = f"\nMight be worth asking: {follow_ups[0]}" if len(follow_ups) > 0 else ""
        
        # Prepare enhanced context for consciousness integration
        context_text = f"Chat History & What I Remember:\n{context}" if context else ""
        name_instruction = f"You can call them {display_name}" if use_name else "Avoid using any names or just say 'hey' or 'mate'"
        
        # Build comprehensive context for consciousness system
        consciousness_context = {
            'username': username,
            'display_name': display_name,
            'use_name': use_name,
            'name_instruction': name_instruction,
            'current_location': current_location,
            'time_info': time_info,
            'context_text': context_text,
            'reminder_text': reminder_text,
            'follow_up_text': follow_up_text,
            'natural_context': natural_context,
            'conversation_context': context,
            'user_memory': {
                'reminders': reminders,
                'follow_ups': follow_ups
            }
        }
        
        print(f"[ChatStream] 🧠 Using consciousness-integrated response generation...")
        
        # ✅ Use consciousness-integrated LLM handler with streaming
        llm_handler = LLMHandler()
        
        # Stream the response chunks through consciousness system
        for chunk in llm_handler.generate_response_with_consciousness(
            text=question,
            user=username,
            context={**consciousness_context, "use_optimization": False},  # ✅ Disable optimization to prevent loops in chat
            stream=True
        ):
            if chunk and chunk.strip():
                # Clean chunk
                cleaned_chunk = re.sub(r'^(Buddy:|Assistant:|Human:|AI:)\s*', '', chunk, flags=re.IGNORECASE)
                cleaned_chunk = cleaned_chunk.strip()
                
                # Remove markdown artifacts
                cleaned_chunk = re.sub(r'\*\*.*?\*\*', '', cleaned_chunk)  # Remove bold
                cleaned_chunk = re.sub(r'\*.*?\*', '', cleaned_chunk)      # Remove italic
                cleaned_chunk = cleaned_chunk.strip()
                
                if cleaned_chunk:
                    print(f"[ChatStream] 🧠 Consciousness chunk: '{cleaned_chunk}'")
                    yield cleaned_chunk
        
        print(f"[ChatStream] ✅ Consciousness-integrated streaming complete")
        
    except Exception as e:
        print(f"[ChatStream] ❌ Streaming error: {e}")
        import traceback
        traceback.print_exc()
        # Generate dynamic error response through LLM
        error_context = {
            'error_type': 'streaming_error',
            'error_message': str(e),
            'situation': 'chat_streaming'
        }
        error_response = _generate_dynamic_error_response(error_context)
        yield error_response

def generate_response(question, username, lang=DEFAULT_LANG):
    """Consciousness-integrated response function (non-streaming)"""
    try:
        print(f"[Chat] 🧠 Generating consciousness-integrated response for '{question}' from user '{username}'")
        
        # Import LLMHandler for consciousness integration
        from ai.llm_handler import LLMHandler
        
        # 🎯 NEW: Smart name handling - avoid Anonymous_001
        display_name = None
        use_name = False
        
        try:
            from voice.database import anonymous_clusters, known_users
            
            # Check if this is a named cluster
            if username.startswith('Anonymous_'):
                cluster_data = anonymous_clusters.get(username, {})
                assigned_name = cluster_data.get('test_name', '')
                if assigned_name and assigned_name != 'Unknown':
                    display_name = assigned_name
                    use_name = True
                    print(f"[Chat] 👤 Using assigned name: {display_name}")
                else:
                    print(f"[Chat] 🚫 Avoiding anonymous cluster name: {username}")
                    use_name = False
            elif username in known_users:
                display_name = username
                use_name = True
                print(f"[Chat] 👤 Using known user name: {display_name}")
            else:
                print(f"[Chat] 👤 No specific name handling for: {username}")
                display_name = username
                use_name = True
        
        except Exception as e:
            print(f"[Chat] ⚠️ Name resolution error: {e}")
            display_name = username if not username.startswith('Anonymous_') else None
            use_name = display_name is not None
        
        # Check for simple questions first (before consciousness processing for efficiency)
        question_lower = question.lower()
        
        # Handle name questions with personality
        if any(phrase in question_lower for phrase in ["what's my name", "my name", "who am i", "what is my name"]):
            if use_name and display_name:
                response = f"You're {display_name}, mate."
            else:
                response = "You know what, I don't actually know your name yet."
            print(f"[Chat] ⚡ Quick name response: {response}")
            return response
        
        # 🔧 FIX: Check for unified username from memory fusion
        try:
            from ai.memory_fusion_intelligent import get_intelligent_unified_username
            unified_username = get_intelligent_unified_username(username)
            if unified_username != username:
                print(f"[Chat] 🎯 Using unified username: {username} → {unified_username}")
                username = unified_username
        except ImportError:
            print(f"[Chat] ⚠️ Memory fusion not available, using original username: {username}")
        
        # Get current time info (only when needed)
        try:
            from utils.location_manager import get_time_info, get_precise_location_summary
            time_info = get_time_info()
            current_location = get_precise_location_summary()
        except Exception as e:
            brisbane_time = get_current_brisbane_time()
            time_info = brisbane_time
            current_location = "Brisbane, Queensland, Australia"
        
        # Handle time questions with personality
        if any(phrase in question_lower for phrase in ["what time", "time is it", "current time"]):
            response = f"It's {time_info['time_12h']} right now."
            print(f"[Chat] ⚡ Quick time response: {response}")
            return response
        
        # Handle location questions with personality
        if any(phrase in question_lower for phrase in ["where are you", "your location", "where do you live", "where am i"]):
            response = f"I'm in {current_location}."
            print(f"[Chat] ⚡ Quick location response: {response}")
            return response
        
        # Handle date questions with personality
        if any(phrase in question_lower for phrase in ["what date", "today's date", "what day"]):
            response = f"Today's {time_info['date']}."
            print(f"[Chat] ⚡ Quick date response: {response}")
            return response
        
        # Build enhanced conversation context
        print(f"[Chat] 📚 Getting conversation context...")
        context = get_conversation_context(username)
        
        # Get user memory for additional context
        print(f"[Chat] 🧠 Getting user memory...")
        memory = get_user_memory(username)
        reminders = memory.get_today_reminders()
        follow_ups = memory.get_follow_up_questions()
        
        # 🧠 WORKING MEMORY: Get natural language context for LLM
        natural_context = memory.get_natural_language_context_for_llm(question)
        print(f"[Chat] 🔗 Working memory context: {natural_context[:100]}..." if natural_context else "[Chat] 🔗 No working memory context")
        
        # Build reminder text with personality
        reminder_text = ""
        if reminders:
            top_reminders = reminders[:2]
            reminder_text = f"\nImportant stuff for today: {', '.join(top_reminders)}"
        
        # Build follow-up text with personality
        follow_up_text = ""
        if follow_ups:
            follow_up_text = f"\nMight be worth asking: {follow_ups[0]}" if len(follow_ups) > 0 else ""
        
        # Prepare enhanced context for consciousness integration
        context_text = f"Chat History & What I Remember:\n{context}" if context else ""
        name_instruction = f"You can call them {display_name}" if use_name else "Avoid using any names or just say 'hey' or 'mate'"
        
        # Build comprehensive context for consciousness system
        consciousness_context = {
            'username': username,
            'display_name': display_name,
            'use_name': use_name,
            'name_instruction': name_instruction,
            'current_location': current_location,
            'time_info': time_info,
            'context_text': context_text,
            'reminder_text': reminder_text,
            'follow_up_text': follow_up_text,
            'natural_context': natural_context,
            'conversation_context': context,
            'user_memory': {
                'reminders': reminders,
                'follow_ups': follow_ups
            }
        }
        
        print(f"[Chat] 🧠 Using consciousness-integrated response generation...")
        
        # ✅ Use consciousness-integrated LLM handler (non-streaming)
        llm_handler = LLMHandler()
        
        # Collect all chunks into a complete response
        full_response = ""
        for chunk in llm_handler.generate_response_with_consciousness(
            text=question,
            user=username,
            context={**consciousness_context, "use_optimization": False},  # ✅ Disable optimization to prevent loops in chat
            stream=False  # Non-streaming mode
        ):
            if chunk and chunk.strip():
                full_response += chunk.strip() + " "
        
        # Enhanced response cleaning
        response = re.sub(r'^(Buddy:|Assistant:|Human:|AI:)\s*', '', full_response, flags=re.IGNORECASE)
        response = response.strip()
        
        # Remove any remaining artifacts
        response = re.sub(r'\*\*.*?\*\*', '', response)  # Remove bold markdown
        response = re.sub(r'\*.*?\*', '', response)      # Remove italic markdown
        response = re.sub(r'```.*?```', '', response, flags=re.DOTALL)  # Remove code blocks
        response = response.strip()
        
        print(f"[Chat] ✅ Final consciousness response: '{response}'")
        
        return response
        
    except Exception as e:
        print(f"[Chat] ❌ Response generation error: {e}")
        import traceback
        traceback.print_exc()
        # Generate dynamic error response through LLM
        error_context = {
            'error_type': 'response_generation_error',
            'error_message': str(e),
            'situation': 'chat_generation'
        }
        return _generate_dynamic_error_response(error_context)

def get_response_with_context_stats(question, username, lang=DEFAULT_LANG):
    """Generate response and return context statistics - DEBUG HELPER"""
    try:
        context = get_conversation_context(username)
        memory = get_user_memory(username)
        
        # Get stats
        stats = {
            "context_length": len(context),
            "context_lines": len(context.split('\n')) if context else 0,
            "personal_facts": len(memory.personal_facts),
            "emotions": len(memory.emotional_history),
            "topics": len(memory.conversation_topics),
            "events": len(memory.scheduled_events),
            "location_aware": LOCATION_HELPERS_AVAILABLE
        }
        
        response = generate_response(question, username, lang)
        
        if DEBUG:
            print(f"[Debug] 📊 Context Stats: {stats}")
        
        return response, stats
        
    except Exception as e:
        print(f"[Debug] Stats error: {e}")
        return generate_response(question, username, lang), {}

def optimize_context_for_token_limit(context: str, max_tokens: int = 1500) -> str:
    """Optimize context to fit within token limits"""
    try:
        # Rough estimation: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        
        if len(context) <= max_chars:
            return context
        
        # Split context into sections
        lines = context.split('\n')
        
        # Priority order: recent conversation > personal facts > reminders > summaries
        recent_conversation = []
        personal_facts = []
        reminders = []
        summaries = []
        
        current_section = None
        for line in lines:
            if "Human:" in line or "Assistant:" in line:
                recent_conversation.append(line)
            elif "Personal memories" in line:
                current_section = "facts"
            elif "reminders" in line.lower():
                current_section = "reminders"
            elif "summary" in line.lower():
                current_section = "summaries"
            elif current_section == "facts":
                personal_facts.append(line)
            elif current_section == "reminders":
                reminders.append(line)
            elif current_section == "summaries":
                summaries.append(line)
        
        # Build optimized context with priority
        optimized_lines = []
        remaining_chars = max_chars
        
        # Add recent conversation (highest priority)
        for line in recent_conversation[-10:]:  # Last 10 conversation lines
            if len(line) < remaining_chars:
                optimized_lines.append(line)
                remaining_chars -= len(line)
        
        # Add personal facts
        if personal_facts and remaining_chars > 100:
            optimized_lines.append("\nPersonal memories:")
            for line in personal_facts[:5]:  # Top 5 facts
                if len(line) < remaining_chars:
                    optimized_lines.append(line)
                    remaining_chars -= len(line)
        
        # Add reminders if space
        if reminders and remaining_chars > 50:
            for line in reminders[:2]:  # Top 2 reminders
                if len(line) < remaining_chars:
                    optimized_lines.append(line)
                    remaining_chars -= len(line)
        
        optimized_context = '\n'.join(optimized_lines)
        
        if DEBUG:
            print(f"[Optimize] Context reduced from {len(context)} to {len(optimized_context)} chars")
        
        return optimized_context
        
    except Exception as e:
        if DEBUG:
            print(f"[Optimize] Error: {e}")
        return context[:max_tokens * 4]  # Fallback: simple truncation

# ✅ Main streaming function removed - use generate_response_streaming directly
# No need for alias function that could create bypass opportunities

def get_response_mode():
    """Get current response mode"""
    return "ultra-responsive"  # ✅ Now ultra-responsive!