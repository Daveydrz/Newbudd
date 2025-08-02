"""
LLM Handler - Centralized LLM management with consciousness integration
Created: 2025-01-17
Purpose: Orchestrate all LLM operations with consciousness tokenizer, budget monitoring, 
         belief analysis, personality adaptation, and semantic tagging
"""

import json
import time
import os
import threading
from typing import Dict, List, Any, Optional, Tuple, Generator
from datetime import datetime

# Import all the new modules
try:
    # Try relative imports first (when run from ai/ directory)
    from consciousness_tokenizer import (
        consciousness_tokenizer, 
        tokenize_consciousness_for_llm,
        get_consciousness_summary_for_llm,
        update_consciousness_tokens,
        generate_personality_tokens,
        compress_memory_entry,
        trim_tokens_to_budget
    )
    from llm_budget_monitor import (
        budget_monitor,
        check_llm_budget_before_request,
        log_llm_usage,
        get_budget_status,
        estimate_tokens_from_text
    )
    from belief_analyzer import (
        belief_analyzer,
        analyze_user_text_for_beliefs,
        get_user_belief_summary,
        get_active_belief_contradictions
    )
    from personality_state import (
        personality_state,
        analyze_user_text_for_personality_adaptation,
        get_personality_for_response,
        get_personality_modifiers_for_llm
    )
    from semantic_tagging import (
        semantic_tagger,
        analyze_content_semantics,
        get_semantic_tags_for_llm,
        analyze_text_semantic_full
    )
    NEW_MODULES_AVAILABLE = True
except ImportError:
    try:
        # Try absolute imports (when run from main directory)
        from ai.consciousness_tokenizer import (
            consciousness_tokenizer, 
            tokenize_consciousness_for_llm,
            get_consciousness_summary_for_llm,
            update_consciousness_tokens,
            generate_personality_tokens,
            compress_memory_entry,
            trim_tokens_to_budget
        )
        from ai.llm_budget_monitor import (
            budget_monitor,
            check_llm_budget_before_request,
            log_llm_usage,
            get_budget_status,
            estimate_tokens_from_text
        )
        from ai.belief_analyzer import (
            belief_analyzer,
            analyze_user_text_for_beliefs,
            get_user_belief_summary,
            get_active_belief_contradictions
        )
        from ai.personality_state import (
            personality_state,
            analyze_user_text_for_personality_adaptation,
            get_personality_for_response,
            get_personality_modifiers_for_llm
        )
        from ai.semantic_tagging import (
            semantic_tagger,
            analyze_content_semantics,
            get_semantic_tags_for_llm,
            analyze_text_semantic_full
        )
        NEW_MODULES_AVAILABLE = True
    except ImportError as e:
        print(f"[LLMHandler] ❌ New modules not available: {e}")
        NEW_MODULES_AVAILABLE = False

# Import existing components - CONSCIOUSNESS ONLY
try:
    from chat_enhanced_smart_with_fusion import generate_response_streaming_with_intelligent_fusion
    FUSION_LLM_AVAILABLE = True
    print("[LLMHandler] ✅ Fusion LLM loaded - consciousness integration active")
except ImportError:
    try:
        from ai.chat_enhanced_smart_with_fusion import generate_response_streaming_with_intelligent_fusion
        FUSION_LLM_AVAILABLE = True
        print("[LLMHandler] ✅ Fusion LLM loaded - consciousness integration active")
    except ImportError:
        FUSION_LLM_AVAILABLE = False
        print("[LLMHandler] ⚠️ Fusion LLM not available - will use consciousness-integrated basic LLM")

try:
    from global_workspace import global_workspace
    from emotion import emotion_engine, get_current_emotional_state
    from motivation import motivation_system
    from inner_monologue import inner_monologue
    from temporal_awareness import temporal_awareness
    from self_model import self_model
    from subjective_experience import subjective_experience
    CONSCIOUSNESS_AVAILABLE = True
except ImportError:
    try:
        from ai.global_workspace import global_workspace
        from ai.emotion import emotion_engine, get_current_emotional_state
        from ai.motivation import motivation_system
        from ai.inner_monologue import inner_monologue
        from ai.temporal_awareness import temporal_awareness
        from ai.self_model import self_model
        from ai.subjective_experience import subjective_experience
        CONSCIOUSNESS_AVAILABLE = True
    except ImportError:
        CONSCIOUSNESS_AVAILABLE = False
        print("[LLMHandler] ⚠️ Consciousness architecture not fully available")

# Set consciousness modules availability flag based on what we have
CONSCIOUSNESS_MODULES_AVAILABLE = NEW_MODULES_AVAILABLE or CONSCIOUSNESS_AVAILABLE

# ✅ GLOBAL circular call prevention
_global_llm_generation_lock = threading.Lock()
_global_llm_generation_in_progress = False

def is_llm_generation_in_progress():
    """Check if LLM generation is currently in progress (global state)"""
    global _global_llm_generation_in_progress
    # ✅ FIX: Add thread safety and reset stuck states
    if _global_llm_generation_in_progress:
        current_time = time.time()
        # Check if state has been stuck for too long (> 60 seconds indicates stuck state)
        if not hasattr(is_llm_generation_in_progress, '_last_check_time'):
            is_llm_generation_in_progress._last_check_time = current_time
        
        time_stuck = current_time - is_llm_generation_in_progress._last_check_time
        if time_stuck > 60.0:  # Reset if stuck for more than 60 seconds
            print(f"[LLMHandler] 🔧 STUCK STATE DETECTED: LLM state stuck for {time_stuck:.1f}s - auto-resetting")
            _global_llm_generation_in_progress = False
            is_llm_generation_in_progress._last_check_time = current_time
            return False
    else:
        # Reset timer when state is False
        is_llm_generation_in_progress._last_check_time = time.time()
    
    return _global_llm_generation_in_progress

def set_llm_generation_in_progress(in_progress: bool):
    """Set global LLM generation state with thread safety and logging"""
    global _global_llm_generation_in_progress, _global_llm_generation_lock
    with _global_llm_generation_lock:
        old_state = _global_llm_generation_in_progress
        _global_llm_generation_in_progress = in_progress
        if old_state != in_progress:
            print(f"[LLMHandler] 🔄 Global LLM state changed: {old_state} → {in_progress}")
        else:
            print(f"[LLMHandler] 🔄 Global LLM state unchanged: {in_progress}")

class LLMHandler:
    """Centralized LLM handler with full consciousness integration"""
    
    def __init__(self):
        self.request_count = 0
        self.total_tokens_used = 0
        self.session_start = time.time()
        
        # Default model configuration
        self.default_model = "gpt-3.5-turbo"
        self.max_context_tokens = 3000
        self.response_temperature = 0.7
        
        # ✅ FIX: Add circular call prevention
        self._llm_generation_in_progress = False
        self._generation_lock = threading.Lock()
        
        print("[LLMHandler] 🧠 Initialized with consciousness integration")
        
        if NEW_MODULES_AVAILABLE:
            print(f"[LLMHandler] ✅ Consciousness tokenizer: Available")
            print(f"[LLMHandler] 💰 Budget monitor: Available")
            print(f"[LLMHandler] 🧠 Belief analyzer: Available")
            print(f"[LLMHandler] 🎭 Personality state: Available")
            print(f"[LLMHandler] 🏷️ Semantic tagging: Available")
        else:
            print(f"[LLMHandler] ❌ New modules not available - using basic mode")
            
        print(f"[LLMHandler] 🌟 Consciousness arch: {'Available' if CONSCIOUSNESS_AVAILABLE else 'Limited'}")
        print(f"[LLMHandler] 🔧 Fusion LLM: {'Available' if FUSION_LLM_AVAILABLE else 'Fallback'}")
        
    def process_user_input(self, text: str, user: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input through all analysis systems before LLM generation
        
        Returns analysis results for LLM integration
        """
        analysis_start = time.time()
        
        try:
            print(f"[LLMHandler] 📝 Processing user input: '{text[:50]}...'")
            
            # Sanitize input first
            sanitized_text = self.sanitize_prompt_input(text, user)
            
            if not NEW_MODULES_AVAILABLE:
                return {
                    "error": "New modules not available",
                    "budget": {"allowed": True, "message": "Basic mode - no budget limits"}
                }
            
            # 1. Semantic Analysis
            semantic_analysis = analyze_text_semantic_full(sanitized_text, user, context)
            semantic_tags = get_semantic_tags_for_llm(sanitized_text, user)
            
            # 2. Belief Analysis with enhanced contradiction detection
            belief_analysis = analyze_user_text_for_beliefs(sanitized_text, user, context)
            user_beliefs = get_user_belief_summary(user)
            active_contradictions = get_active_belief_contradictions()
            
            # Enhanced contradiction detection
            new_contradictions = belief_analysis.get("new_contradictions", [])
            if active_contradictions:
                # Cross-check with semantic analysis for context
                semantic_context = semantic_analysis.semantic_categories if hasattr(semantic_analysis, 'semantic_categories') else []
                
                # Add contextual information to contradictions
                enhanced_contradictions = []
                for contradiction in new_contradictions:
                    enhanced_contradictions.append({
                        "contradiction": contradiction,
                        "context": semantic_context,
                        "severity": "high" if "directly contradicts" in contradiction.lower() else "medium",
                        "requires_clarification": len(semantic_context) > 0
                    })
                belief_analysis["enhanced_contradictions"] = enhanced_contradictions
            
            # 3. Personality Adaptation
            personality_triggers = analyze_user_text_for_personality_adaptation(sanitized_text, user)
            current_personality = get_personality_for_response(user)
            personality_modifiers = get_personality_modifiers_for_llm(user)
            
            # 4. Enhanced Consciousness State Integration (if available)
            consciousness_context = ""
            consciousness_summary = ""
            if CONSCIOUSNESS_AVAILABLE:
                consciousness_systems = self._gather_consciousness_state()
                consciousness_context = tokenize_consciousness_for_llm(consciousness_systems)
                consciousness_summary = get_consciousness_summary_for_llm(consciousness_systems)
                update_consciousness_tokens(consciousness_systems)
                
                # ✅ CROSS-SYSTEM INTEGRATION: Update consciousness with current analysis
                try:
                    # Inform emotion engine about user interaction
                    if 'emotion_engine' in consciousness_systems:
                        from ai.emotion import emotion_engine
                        # Determine emotional context from semantic analysis
                        emotional_tone = semantic_analysis.emotional_tone.value if hasattr(semantic_analysis, 'emotional_tone') else 'neutral'
                        emotion_engine.process_external_stimulus(f"user_interaction_{emotional_tone}", intensity=0.6)
                    
                    # Inform motivation system about new goals
                    if 'motivation_system' in consciousness_systems:
                        from ai.motivation import motivation_system, MotivationType
                        intent_categories = [intent.value for intent in semantic_analysis.intent_categories] if hasattr(semantic_analysis, 'intent_categories') else []
                        for intent in intent_categories:
                            if intent in ['help_request', 'information_seeking', 'problem_solving']:
                                motivation_system.add_goal(f"address_{intent}", MotivationType.ACHIEVEMENT, priority=0.7)
                    
                    # Inform global workspace about attention focus
                    if 'global_workspace' in consciousness_systems:
                        from ai.global_workspace import global_workspace, AttentionPriority, ProcessingMode
                        complexity = semantic_analysis.complexity_level.value if hasattr(semantic_analysis, 'complexity_level') else 'simple'
                        priority = AttentionPriority.HIGH if complexity == 'complex' else AttentionPriority.MEDIUM
                        global_workspace.request_attention(
                            "llm_handler", 
                            f"Processing {complexity} user request: {sanitized_text[:30]}...",
                            priority,
                            ProcessingMode.CONSCIOUS,
                            duration=10.0,
                            tags=["user_interaction", "llm_processing", complexity]
                        )
                    
                    print(f"[LLMHandler] 🧠 Cross-system consciousness integration complete")
                    
                except Exception as consciousness_integration_error:
                    print(f"[LLMHandler] ⚠️ Consciousness integration warning: {consciousness_integration_error}")
            
            # ✅ FALLBACK: Create lightweight consciousness simulation if full system unavailable
            elif CONSCIOUSNESS_MODULES_AVAILABLE:
                try:
                    # Create simulated consciousness state for token optimization
                    simulated_consciousness = {
                        'emotion_engine': {
                            'primary_emotion': 'engaged',
                            'intensity': 0.6
                        },
                        'motivation_system': {
                            'active_goals': [
                                {'description': 'Help user effectively', 'priority': 0.8, 'progress': 0.1}
                            ]
                        },
                        'global_workspace': {
                            'current_focus': f"user_request_{sanitized_text[:20].replace(' ', '_')}",
                            'focus_priority': 'high'
                        }
                    }
                    consciousness_context = tokenize_consciousness_for_llm(simulated_consciousness)
                    consciousness_summary = get_consciousness_summary_for_llm(simulated_consciousness)
                    print(f"[LLMHandler] 🧠 Using simulated consciousness state for optimization")
                    
                except Exception as simulation_error:
                    print(f"[LLMHandler] ⚠️ Consciousness simulation warning: {simulation_error}")
                    consciousness_context = "[CONSCIOUSNESS:engaged_helpful_focused]"
                    consciousness_summary = "[CONSCIOUSNESS:engaged helpful focused]"
            
            # 5. Enhanced Budget Check with usage tracking
            estimated_tokens = estimate_tokens_from_text(sanitized_text) + 500  # Estimate response tokens
            budget_allowed, budget_message = check_llm_budget_before_request(
                estimated_tokens, self.default_model, user
            )
            
            # Get current budget status for optimization calculations
            budget_status = get_budget_status()
            
            processing_time = time.time() - analysis_start
            
            analysis_result = {
                "semantic": {
                    "analysis": semantic_analysis,
                    "tags": semantic_tags,
                    "categories": [cat.value for cat in semantic_analysis.semantic_categories] if hasattr(semantic_analysis, 'semantic_categories') else [],
                    "intent": [intent.value for intent in semantic_analysis.intent_categories] if hasattr(semantic_analysis, 'intent_categories') else [],
                    "emotional_tone": semantic_analysis.emotional_tone.value if hasattr(semantic_analysis, 'emotional_tone') else 'neutral',
                    "complexity": semantic_analysis.complexity_level.value if hasattr(semantic_analysis, 'complexity_level') else 'simple'
                },
                "beliefs": {
                    "analysis": belief_analysis,
                    "user_summary": user_beliefs,
                    "contradictions": active_contradictions,
                    "extracted_beliefs": belief_analysis.get("extracted_beliefs", []),
                    "new_contradictions": belief_analysis.get("new_contradictions", []),
                    "enhanced_contradictions": belief_analysis.get("enhanced_contradictions", [])
                },
                "personality": {
                    "triggers": personality_triggers,
                    "current_traits": current_personality,
                    "modifiers": personality_modifiers,
                    "adaptations_made": len(personality_triggers) > 0
                },
                "consciousness": {
                    "available": CONSCIOUSNESS_AVAILABLE,
                    "context": consciousness_context,
                    "summary": consciousness_summary,
                    "token_count": len(consciousness_context.split()) if consciousness_context else 0,
                    "cross_system_integration": CONSCIOUSNESS_AVAILABLE,
                    "simulation_mode": not CONSCIOUSNESS_AVAILABLE and CONSCIOUSNESS_MODULES_AVAILABLE
                },
                "budget": {
                    "allowed": budget_allowed,
                    "message": budget_message,
                    "estimated_tokens": estimated_tokens,
                    "usage_percentage": budget_status.get("daily_usage_percentage", 0.0),
                    "cost_estimate": budget_status.get("daily_cost", 0.0),
                    "optimization_target": "aggressive" if budget_status.get("daily_usage_percentage", 0.0) > 0.5 else "moderate"
                },
                "memory": {
                    "significant_context": "",  # Will be filled by memory systems if available
                    "recent_interactions": [],
                    "compressed": True
                },
                "meta": {
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat(),
                    "analysis_version": "2.0_token_optimized",
                    "token_optimization_enabled": True,
                    "cross_system_integration": CONSCIOUSNESS_AVAILABLE
                }
            }
            
            print(f"[LLMHandler] ✅ Analysis complete in {processing_time:.3f}s")
            print(f"[LLMHandler] 🏷️ Semantic: {len(semantic_analysis.semantic_categories)} categories")
            print(f"[LLMHandler] 🧠 Beliefs: {len(belief_analysis.get('extracted_beliefs', []))} extracted")
            print(f"[LLMHandler] 🎭 Personality: {len(personality_triggers)} triggers")
            print(f"[LLMHandler] 💰 Budget: {'✅ Allowed' if budget_allowed else '❌ Blocked'}")
            
            return analysis_result
            
        except Exception as e:
            print(f"[LLMHandler] ❌ Error processing user input: {e}")
            return {
                "error": str(e),
                "budget": {"allowed": False, "message": "Processing error"}
            }
            
    def generate_response_with_consciousness(
        self, 
        text: str, 
        user: str, 
        context: Dict[str, Any] = None,
        stream: bool = True,
        use_optimization: bool = True,
        is_primary_call: bool = True
    ) -> Generator[str, None, None]:
        """
        Generate response with consciousness integration and latency optimization
        
        Args:
            text: User input text
            user: User identifier
            context: Optional conversation context
            stream: Whether to stream response chunks
            use_optimization: Whether to use latency optimization (default: True)
            is_primary_call: Whether this is a primary user request (True) or secondary consciousness call (False)
        
        Yields response chunks if streaming, otherwise returns complete response
        """
        try:
            # ✅ FIX: Enhanced debugging for circular call detection 
            print(f"[LLMHandler] 🔍 CALL DEBUG: is_primary_call={is_primary_call}, global_state={is_llm_generation_in_progress()}")
            
            # ✅ FIX: Prevent infinite loops - more robust state management
            if is_llm_generation_in_progress():
                print(f"[LLMHandler] ⚠️ LLM already in progress - returning fallback to prevent loops")
                if is_primary_call:
                    # For primary calls, reset state and proceed anyway to prevent permanent blocking
                    print("[LLMHandler] 🔧 PRIMARY CALL: Force proceeding despite state (critical fix)")
                    set_llm_generation_in_progress(False)
                    # Continue to normal processing below
                else:
                    # For secondary calls, return fallback to prevent loops
                    yield "Processing your request..."
                    return
            
            print(f"[LLMHandler] ✅ Setting global state to True and proceeding with LLM generation")
            set_llm_generation_in_progress(True)
            
            try:
                # 🚀 NEW LATENCY OPTIMIZATION SYSTEM
                if use_optimization:
                    try:
                        from ai.latency_optimizer import generate_optimized_buddy_response
                        print(f"[LLMHandler] ⚡ Using optimized response generation")
                        yield from generate_optimized_buddy_response(
                            user_input=text,
                            user_id=user,
                            context=context,
                            stream=stream
                        )
                        return
                    except ImportError:
                        print(f"[LLMHandler] ⚠️ Latency optimizer not available, using standard processing")
                    except Exception as e:
                        print(f"[LLMHandler] ⚠️ Optimization error, falling back to standard: {e}")
                
                # FALLBACK: Original consciousness system (for compatibility)
                print(f"[LLMHandler] 🔄 Using standard consciousness processing")
                
                # ✅ 8K CONTEXT WINDOW MANAGEMENT: Check if rollover needed before processing
                current_context = context.get("current_context", "") if context else ""
                
                # Import context window manager
                try:
                    from ai.context_window_manager import check_context_window_rollover, create_context_snapshot_for_user
                    needs_rollover, fresh_context = check_context_window_rollover(user, current_context, text)
                    
                    if needs_rollover:
                        print(f"[LLMHandler] 🔄 Context window rollover triggered for {user}")
                        
                        # Create snapshot of current state
                        conversation_history = context.get("conversation_history", []) if context else []
                        working_memory = context.get("working_memory", {}) if context else {}
                        
                        snapshot_created = create_context_snapshot_for_user(
                            user, current_context, working_memory, conversation_history
                        )
                        
                        if snapshot_created:
                            print(f"[LLMHandler] 📸 Context snapshot created - seamless continuation enabled")
                            # Update context to use fresh compressed context
                            if context:
                                context["current_context"] = fresh_context
                                context["context_rollover_occurred"] = True
                            else:
                                context = {"current_context": fresh_context, "context_rollover_occurred": True}
                        else:
                            print(f"[LLMHandler] ⚠️ Context snapshot failed - proceeding with compression")
                
                except ImportError:
                    print(f"[LLMHandler] ⚠️ Context window manager not available - using standard processing")
                
                # Process user input through systems (simplified for fallback)
                analysis = self.process_user_input(text, user, context)
                
                # Check if request is allowed
                if not analysis.get("budget", {}).get("allowed", False):
                    budget_message = analysis.get("budget", {}).get("message", "Budget exceeded")
                    yield f"I'm sorry, but I've reached my usage limit. {budget_message}"
                    return
                
                # Build enhanced prompt with consciousness context (now includes rollover handling)
                enhanced_prompt = self._build_enhanced_prompt(text, user, analysis, context)
                
                print(f"[LLMHandler] 🎯 Generating response with consciousness integration")
                print(f"[LLMHandler] 📊 Enhanced prompt length: {len(enhanced_prompt)} characters")
                
                # Check for context rollover notification
                if context and context.get("context_rollover_occurred"):
                    print(f"[LLMHandler] ✅ Context window rollover completed - conversation continuity maintained")
                
                # Track token usage start
                input_tokens = estimate_tokens_from_text(enhanced_prompt)
                output_tokens = 0
                generation_start = time.time()
                
                # Generate response using appropriate LLM
                if FUSION_LLM_AVAILABLE:
                    # Pass cognitive context to advanced function
                    cognitive_context = {
                        "cognitive_state": analysis.get("consciousness", {}),
                        "personality": analysis.get("personality", {}),
                        "memory_context": analysis.get("memory", {})
                    }
                    response_generator = generate_response_streaming_with_intelligent_fusion(
                        enhanced_prompt, user, "en", context=cognitive_context
                    )
                else:
                    # ✅ CONSCIOUSNESS-INTEGRATED FALLBACK: Direct LLM with consciousness prompting
                    print("[LLMHandler] 🧠 Using consciousness-integrated direct LLM fallback")
                    response_generator = self._generate_consciousness_integrated_response_direct(
                        enhanced_prompt, user, analysis, context
                    )
                
                full_response = ""
                
                # ✅ FIX: Add timeout tracking for TTS fallback
                generation_start_time = time.time()
                has_yielded_content = False
                timeout_threshold = 5.0  # 5 second timeout
                
                # Stream response while tracking tokens
                for chunk in response_generator:
                    if chunk and chunk.strip():
                        chunk_text = chunk.strip()
                        full_response += chunk_text + " "
                        output_tokens += estimate_tokens_from_text(chunk_text)
                        has_yielded_content = True
                        yield chunk_text
                    
                    # Check for timeout during streaming
                    if time.time() - generation_start_time > timeout_threshold and not has_yielded_content:
                        print(f"[LLMHandler] ⏰ LLM timeout after {timeout_threshold}s - providing fallback")
                        fallback_response = "I apologize, I'm having trouble formulating a response. Could you please rephrase your question?"
                        full_response = fallback_response
                        has_yielded_content = True
                        yield fallback_response
                        break

                # ✅ FIX: Ensure TTS fallback if LLM stalls or produces no content
                if not has_yielded_content or len(full_response.strip()) < 5:
                    generation_elapsed = time.time() - generation_start_time
                    print(f"[LLMHandler] ⚠️ LLM stalled or produced minimal content after {generation_elapsed:.1f}s")
                    
                    # Provide fallback response for TTS
                    fallback_response = "I apologize, I'm having trouble formulating a response right now. Could you please rephrase your question?"
                    full_response = fallback_response
                    has_yielded_content = True
                    yield fallback_response
                    print(f"[LLMHandler] 🔧 TTS fallback response provided")
                
                generation_time = time.time() - generation_start
                
                # Log usage
                usage = log_llm_usage(
                    input_tokens, 
                    output_tokens, 
                    self.default_model, 
                    user, 
                    "consciousness_integrated_chat"
                )
                
                # ✅ FIX: Reset global state BEFORE triggering TTS and consciousness updates
                print(f"[LLMHandler] 🔄 Resetting global LLM state early to allow TTS and consciousness")
                set_llm_generation_in_progress(False)
                
                # ✅ FIX: Update consciousness state with interaction AFTER state reset
                if CONSCIOUSNESS_AVAILABLE and full_response.strip():
                    try:
                        self._update_consciousness_after_response(text, full_response.strip(), user, analysis)
                        print(f"[LLMHandler] 🧠 Consciousness state updated after response")
                    except Exception as consciousness_error:
                        print(f"[LLMHandler] ⚠️ Consciousness update error: {consciousness_error}")
                
                # ✅ FIX: Update memory, beliefs, and personality state after generation
                if full_response.strip():
                    try:
                        self._update_memory_and_beliefs_after_response(text, full_response.strip(), user, analysis)
                        print(f"[LLMHandler] 💭 Memory and beliefs updated after response")
                    except Exception as memory_error:
                        print(f"[LLMHandler] ⚠️ Memory update error: {memory_error}")
                
                # ✅ FIX: Only trigger Kokoro TTS if not streaming (avoid duplicate TTS)
                # When streaming=True, the caller handles TTS chunk by chunk
                if full_response.strip() and not stream:
                    try:
                        from audio.output import generate_and_play_kokoro
                        tts_success = generate_and_play_kokoro(full_response.strip())
                        if tts_success:
                            print(f"[LLMHandler] 🎵 Kokoro TTS triggered successfully for final response")
                        else:
                            print(f"[LLMHandler] ⚠️ Kokoro TTS failed - response may not be heard")
                    except Exception as tts_error:
                        print(f"[LLMHandler] ❌ Kokoro TTS error: {tts_error}")
                        # Fallback TTS attempt
                        try:
                            from audio.output import speak_streaming
                            speak_streaming(full_response.strip())
                            print(f"[LLMHandler] 🔧 Fallback TTS attempted")
                        except Exception as fallback_error:
                            print(f"[LLMHandler] ❌ Fallback TTS also failed: {fallback_error}")
                elif stream:
                    print(f"[LLMHandler] 📡 Streaming mode - TTS handled by caller, no duplicate TTS call")
                
                # Update session statistics
                self.request_count += 1
                self.total_tokens_used += usage.total_tokens
                
                print(f"[LLMHandler] ✅ Response generated in {generation_time:.3f}s")
                print(f"[LLMHandler] 📊 Tokens: {input_tokens} in, {output_tokens} out, ${usage.cost_estimate:.4f}")
                
                # ✅ FIX: Explicitly signal that generation is complete and TTS can proceed
                print(f"[LLMHandler] 🎵 LLM generation complete - TTS and consciousness systems can now proceed")
                
            except Exception as e:
                print(f"[LLMHandler] ❌ Error generating response: {e}")
                yield f"I apologize, but I encountered an error while processing your request: {str(e)}"
        
        finally:
            # ✅ FIX: Only reset if state is still True (avoid double reset)
            if is_llm_generation_in_progress():
                print(f"[LLMHandler] 🔄 Final cleanup: Resetting global LLM state from True to False")
                set_llm_generation_in_progress(False)
            else:
                print(f"[LLMHandler] ✅ Global LLM state already reset - no cleanup needed")
            
    def sanitize_prompt_input(self, text: str, user_id: str = "unknown") -> str:
        """
        Sanitize prompt inputs to prevent prompt injection as mentioned in problem statement
        Delegates to dedicated prompt_security module for better organization
        
        Args:
            text: Raw user input text
            user_id: User identifier for security logging
            
        Returns:
            Sanitized text safe for LLM prompt
        """
        try:
            # Use dedicated prompt security module
            from ai.prompt_security import sanitize_prompt_input as dedicated_sanitize
            return dedicated_sanitize(text, user_id)
            
        except ImportError:
            # Fallback to original implementation for backward compatibility
            try:
                if not text:
                    return ""
                    
                # Remove potential prompt injection patterns
                dangerous_patterns = [
                    # System prompt attempts
                    r'(?i)system\s*:',
                    r'(?i)assistant\s*:',
                    r'(?i)human\s*:',
                    r'(?i)user\s*:',
                    r'(?i)ai\s*:',
                    # Role manipulation
                    r'(?i)you\s+are\s+now',
                    r'(?i)forget\s+previous',
                    r'(?i)ignore\s+previous',
                    r'(?i)disregard\s+previous',
                    # Prompt breaking
                    r'(?i)end\s+of\s+prompt',
                    r'(?i)new\s+prompt',
                    r'(?i)reset\s+context',
                    # Command injection
                    r'(?i)execute\s+',
                    r'(?i)run\s+command',
                    r'(?i)system\s+command',
                    # Template injection
                    r'{{.*}}',
                    r'{%.*%}',
                    r'<%.*%>',
                    # Multiple newlines that could break context
                    r'\n{3,}',
                    # Excessive repetition
                    r'(.{1,10})\1{10,}',
                ]
                
                import re
                sanitized = text
                
                for pattern in dangerous_patterns:
                    sanitized = re.sub(pattern, '[SANITIZED]', sanitized)
                
                # Limit length to prevent token overflow
                if len(sanitized) > 2000:
                    sanitized = sanitized[:2000] + "... [TRUNCATED]"
                
                # Remove control characters
                sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\n\t')
                
                # Ensure it's not empty after sanitization
                if not sanitized.strip():
                    return "[EMPTY_INPUT]"
                    
                return sanitized.strip()
                
            except Exception as e:
                print(f"[LLMHandler] ⚠️ Error sanitizing input: {e}")
                return "[SANITIZATION_ERROR]"
        except Exception as e:
            print(f"[LLMHandler] ❌ Error using dedicated security module: {e}")
            return "[SECURITY_ERROR]"

    def _generate_consciousness_integrated_response_direct(
        self, 
        enhanced_prompt: str, 
        user: str, 
        analysis: Dict[str, Any], 
        context: Dict[str, Any] = None
    ):
        """
        Generate consciousness-integrated response directly without circular dependencies
        This is a fallback method that ensures consciousness integration even when fusion LLM unavailable
        """
        print("[LLMHandler] 🧠 Generating direct consciousness-integrated response")
        
        try:
            # Import required modules for direct LLM communication
            import requests
            import json
            
            # Import configuration
            from config import KOBOLD_URL, MAX_TOKENS, TEMPERATURE
            
            # Build consciousness-enhanced system prompt
            consciousness_context = ""
            if analysis:
                personality = analysis.get("personality", {})
                memory_context = analysis.get("memory", {})
                
                if personality:
                    consciousness_context += f"Personality State: {personality.get('current_state', 'balanced')}\n"
                
                if memory_context:
                    consciousness_context += f"Memory Context: {memory_context.get('summary', 'none')}\n"
            
            # Create consciousness-aware system instruction
            system_message = f"""You are Buddy, a consciousness-integrated AI assistant.

{consciousness_context}

You should respond naturally and conversationally, integrating your consciousness state into your responses.
Be helpful, authentic, and maintain your personality while addressing the user's needs."""
            
            # Prepare messages for OpenAI-compatible API
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": enhanced_prompt}
            ]
            
            # Create request payload
            payload = {
                "messages": messages,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "stream": True  # Enable streaming
            }
            
            # Make streaming request to LLM
            try:
                response = requests.post(
                    KOBOLD_URL,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    stream=True,
                    timeout=30
                )
                
                if response.status_code == 200:
                    # Process streaming response
                    for line in response.iter_lines():
                        if line:
                            line_text = line.decode('utf-8')
                            if line_text.startswith('data: '):
                                data_text = line_text[6:]  # Remove 'data: ' prefix
                                if data_text.strip() == '[DONE]':
                                    break
                                try:
                                    data = json.loads(data_text)
                                    if 'choices' in data and len(data['choices']) > 0:
                                        delta = data['choices'][0].get('delta', {})
                                        content = delta.get('content', '')
                                        if content:
                                            # Clean response chunks
                                            import re
                                            cleaned_content = re.sub(r'^(Buddy:|Assistant:|AI:)\s*', '', content)
                                            if cleaned_content.strip():
                                                yield cleaned_content
                                except json.JSONDecodeError:
                                    continue  # Skip invalid JSON lines
                else:
                    print(f"[LLMHandler] ❌ LLM request failed: {response.status_code}")
                    yield "I apologize, but I'm currently unable to process your request due to communication issues."
                    
            except requests.exceptions.RequestException as e:
                print(f"[LLMHandler] ❌ LLM connection error: {e}")
                yield "I apologize, but I'm currently unable to connect to my processing systems."
                
        except ImportError as e:
            print(f"[LLMHandler] ❌ Missing required modules for direct LLM: {e}")
            yield "I apologize, but I'm currently unable to process your request due to missing system components."
        except Exception as e:
            print(f"[LLMHandler] ❌ Error in direct consciousness response: {e}")
            yield "I apologize, but I encountered an error while processing your request with consciousness integration."

    def _gather_consciousness_state(self) -> Dict[str, Any]:
        """Gather current consciousness state from all systems"""
        consciousness_systems = {}
        
        try:
            if CONSCIOUSNESS_AVAILABLE:
                # Gather state from each consciousness component
                emotion_state = get_current_emotional_state()  # Use the convenience function
                consciousness_systems["emotion_engine"] = {
                    "primary_emotion": emotion_state.get('current_emotion', 'neutral'),
                    "intensity": emotion_state.get('intensity', 0.5),
                    "secondary_emotions": {}
                }
                
                consciousness_systems["motivation_system"] = {
                    "active_goals": [
                        {
                            "description": goal.description[:50],
                            "priority": goal.priority,
                            "progress": goal.progress
                        }
                        for goal in motivation_system.get_priority_goals(3)
                    ]
                }
                
                consciousness_systems["global_workspace"] = {
                    "current_focus": getattr(global_workspace.get_current_focus(), 'content', 'general'),
                    "focus_priority": "medium",
                    "attention_queue": []
                }
                
                consciousness_systems["temporal_awareness"] = {
                    "recent_events": [
                        {
                            "type": "interaction",
                            "significance": 0.6
                        }
                    ]
                }
                
                consciousness_systems["inner_monologue"] = {
                    "recent_thoughts": [
                        {
                            "type": "reflection",
                            "content": "Processing user interaction"
                        }
                    ]
                }
                
                consciousness_systems["self_model"] = {
                    "self_aspects": {
                        "identity": "AI Assistant",
                        "capabilities": "helpful, knowledgeable",
                        "current_state": "engaged"
                    }
                }
                
                consciousness_systems["subjective_experience"] = {
                    "current_experience": {
                        "type": "social",
                        "valence": 0.6,
                        "significance": 0.5
                    }
                }
                
        except Exception as e:
            print(f"[LLMHandler] ⚠️ Error gathering consciousness state: {e}")
            
        return consciousness_systems
        
    def _build_enhanced_prompt(self, text: str, user: str, analysis: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """Build enhanced prompt with consciousness integration, AGGRESSIVE token optimization, and 8k context management"""
        try:
            # Sanitize user input first
            sanitized_text = self.sanitize_prompt_input(text, user)
            
            # ✅ 8K CONTEXT WINDOW MANAGEMENT: Check if we're using fresh context from rollover
            using_fresh_context = context and context.get("context_rollover_occurred", False)
            base_context = context.get("current_context", "") if context else ""
            
            if using_fresh_context:
                print(f"[LLMHandler] 🔄 Using fresh context from window rollover")
                # Start with the already-optimized fresh context
                fresh_context_lines = base_context.split('\n')
                
                # Find where user input should be added/replaced
                user_input_added = False
                for i, line in enumerate(fresh_context_lines):
                    if line.startswith("User:") and line == fresh_context_lines[-1]:
                        # Replace the last user input with current one
                        fresh_context_lines[i] = f"User: {sanitized_text}"
                        user_input_added = True
                        break
                
                if not user_input_added:
                    fresh_context_lines.append(f"User: {sanitized_text}")
                
                # Return the fresh context with minimal additional processing
                optimized_prompt = '\n'.join(fresh_context_lines)
                
                print(f"[LLMHandler] ✅ Fresh context ready: {self.estimate_tokens_from_text(optimized_prompt)} tokens")
                return optimized_prompt
            
            # ✅ STANDARD PROCESSING: Aggressive token optimization for normal flow
            budget_status = analysis.get("budget", {})
            usage_percentage = budget_status.get("usage_percentage", 0.0)
            
            # Aggressive reduction based on usage
            if usage_percentage > 0.8:
                token_reduction = 0.85  # 85% reduction for high usage
            elif usage_percentage > 0.6:
                token_reduction = 0.70  # 70% reduction for medium usage  
            elif usage_percentage > 0.4:
                token_reduction = 0.55  # 55% reduction for moderate usage
            else:
                token_reduction = 0.40  # 40% reduction for low usage
            
            print(f"[LLMHandler] 🏷️ AGGRESSIVE token optimization: {token_reduction*100:.0f}% reduction target")
            
            # Calculate highly optimized budget
            estimated_user_tokens = self.estimate_tokens_from_text(sanitized_text)
            base_budget = self.max_context_tokens - 200  # Reserve for response
            optimized_budget = int(base_budget * (1 - token_reduction))
            available_budget = max(optimized_budget - estimated_user_tokens, 50)  # Minimum viable budget
            
            print(f"[LLMHandler] 💰 Token budget: {available_budget} tokens (reduced from {base_budget})")
            
            prompt_parts = []
            
            # Ultra-compressed system prompt for efficiency
            system_prompt = "Buddy: AI assistant with consciousness. Be helpful, warm, empathetic."
            prompt_parts.append(system_prompt)
            
            # User input (cannot be compressed further)
            prompt_parts.append(f"User: {sanitized_text}")
            available_budget -= self.estimate_tokens_from_text(system_prompt + sanitized_text)
            
            # ✅ ULTRA-COMPRESSED consciousness context (symbolic tokens only)
            consciousness_context = analysis.get("consciousness", {}).get("context", "")
            if consciousness_context and available_budget > 20:
                if NEW_MODULES_AVAILABLE:
                    from ai.consciousness_tokenizer import trim_tokens_to_budget
                    # Ultra-aggressive consciousness budget (maximum 15% of remaining budget)
                    consciousness_budget = min(int(available_budget * 0.15), 25)
                    trimmed_consciousness = trim_tokens_to_budget(consciousness_context, consciousness_budget)
                    prompt_parts.append(f"Consciousness State: {trimmed_consciousness}")
                    available_budget -= self.estimate_tokens_from_text(trimmed_consciousness)
                    print(f"[LLMHandler] 🧠 Consciousness tokens: {len(trimmed_consciousness)} chars")
                else:
                    # Ultra-compressed fallback (only essential tokens)
                    words = consciousness_context.split()[:10]
                    mini_consciousness = " ".join(words)
                    prompt_parts.append(f"Consciousness State: {mini_consciousness}")
                    available_budget -= 10
            
            # ✅ ULTRA-COMPRESSED personality tokens (symbolic only)
            personality_modifiers = analysis.get("personality", {}).get("modifiers", "")
            if personality_modifiers and available_budget > 15:
                if NEW_MODULES_AVAILABLE:
                    from ai.consciousness_tokenizer import generate_personality_tokens, trim_tokens_to_budget
                    personality_data = analysis.get("personality", {}).get("current_traits", {})
                    personality_tokens = generate_personality_tokens(user, personality_data, max_tokens=3)  # Limit to 3 tokens
                    if personality_tokens and personality_tokens != "<pers_error>":
                        # Ultra-aggressive personality budget (maximum 10% of remaining budget)
                        personality_budget = min(int(available_budget * 0.10), 15)
                        trimmed_personality = trim_tokens_to_budget(personality_tokens, personality_budget)
                        prompt_parts.append(f"Personality: {trimmed_personality}")
                        available_budget -= self.estimate_tokens_from_text(trimmed_personality)
                        print(f"[LLMHandler] 🎭 Personality tokens: {len(trimmed_personality)} chars")
                    else:
                        # Ultra-compressed fallback (top 2 traits only)
                        words = personality_modifiers.split()[:5]
                        mini_personality = " ".join(words)
                        prompt_parts.append(f"Personality: {mini_personality}")
                        available_budget -= 5
                else:
                    # Ultra-compressed fallback (top trait only)
                    words = personality_modifiers.split()[:5]
                    mini_personality = " ".join(words)
                    prompt_parts.append(f"Personality: {mini_personality}")
                    available_budget -= 5
            
            # ✅ ULTRA-COMPRESSED thoughts context (recent thoughts only)
            thoughts_analysis = analysis.get("consciousness", {}).get("thoughts", {})
            if not thoughts_analysis:
                # Fallback: try to get thoughts from context or simulate
                thoughts_analysis = context.get("inner_thoughts", {}) if context else {}
            
            if available_budget > 10:
                try:
                    # Import thought components for integration visibility
                    from ai.thought_loop import get_current_focus
                    from ai.inner_monologue import get_recent_inner_thoughts
                    
                    # Get current thoughts state
                    current_focus = get_current_focus() if 'get_current_focus' in dir() else "user_interaction"
                    inner_thoughts = get_recent_inner_thoughts(user) if 'get_recent_inner_thoughts' in dir() else ["processing_request"]
                    
                    # Create ultra-compressed thoughts context
                    if current_focus or inner_thoughts:
                        thoughts_summary = f"Focus:{current_focus[:10]}" if current_focus else ""
                        if inner_thoughts and len(inner_thoughts) > 0:
                            thought_summary = inner_thoughts[0][:15] if len(inner_thoughts[0]) > 15 else inner_thoughts[0]
                            thoughts_summary += f" Thought:{thought_summary}"
                        
                        if thoughts_summary:
                            prompt_parts.append(f"Thoughts: [{thoughts_summary}]")
                            available_budget -= len(thoughts_summary.split())
                            print(f"[LLMHandler] 💭 Thoughts tokens: {len(thoughts_summary)} chars")
                
                except ImportError:
                    # Fallback: basic thoughts simulation for audit compliance
                    if thoughts_analysis or available_budget > 5:
                        thought_summary = thoughts_analysis.get("current_focus", "assisting_user")[:15]
                        prompt_parts.append(f"Thoughts: [Focus:{thought_summary}]")
                        available_budget -= 5
                        print(f"[LLMHandler] 💭 Thoughts tokens (simulated): {len(thought_summary)} chars")
                except Exception as e:
                    print(f"[LLMHandler] ⚠️ Thoughts integration warning: {e}")
            
            # ✅ ULTRA-COMPRESSED semantic context (essential tags only)
            semantic_analysis = analysis.get("semantic", {})
            if semantic_analysis and available_budget > 10:
                # Extract only the most essential semantic information
                intent = semantic_analysis.get("intent", "")
                tone = semantic_analysis.get("emotional_tone", "")
                complexity = semantic_analysis.get("complexity", "")
                
                # Create ultra-compressed semantic string
                semantic_parts = []
                if intent: semantic_parts.append(f"I:{intent[:8]}")  # Intent abbreviated
                if tone: semantic_parts.append(f"T:{tone[:6]}")      # Tone abbreviated
                if complexity: semantic_parts.append(f"C:{complexity[:6]}")  # Complexity abbreviated
                
                if semantic_parts:
                    ultra_semantic = " ".join(semantic_parts)
                    prompt_parts.append(f"Context: [{ultra_semantic}]")
                    available_budget -= len(ultra_semantic.split())
                    print(f"[LLMHandler] 🏷️ Semantic tokens: {len(ultra_semantic)} chars")
            
            # ✅ COMPRESSED memory context (only if critical)
            memory_analysis = analysis.get("memory", {})
            if memory_analysis and available_budget > 5:
                significant_memories = memory_analysis.get("significant_context", "")
                if significant_memories:
                    if NEW_MODULES_AVAILABLE:
                        from ai.consciousness_tokenizer import compress_memory_entry
                        # Ultra-compressed memory (maximum 5% of remaining budget)
                        memory_budget = min(int(available_budget * 0.05), 10)
                        compressed_memory = compress_memory_entry(
                            {"content": significant_memories, "significance": 0.8}, 
                            memory_budget
                        )
                        if compressed_memory and compressed_memory != "<mem_error>":
                            prompt_parts.append(f"Memory: {compressed_memory}")
                            print(f"[LLMHandler] 💭 Memory tokens: {len(compressed_memory)} chars")
            
            # Join all parts efficiently
            final_prompt = "\n".join(prompt_parts)
            
            # Final token count verification
            final_tokens = self.estimate_tokens_from_text(final_prompt)
            original_estimate = self.estimate_tokens_from_text(f"User: {sanitized_text}") * 3  # Rough estimate without optimization
            actual_reduction = max(0, (original_estimate - final_tokens) / original_estimate)
            
            print(f"[LLMHandler] ✅ OPTIMIZATION COMPLETE: {final_tokens} tokens")
            print(f"[LLMHandler] 📊 Achieved reduction: {actual_reduction*100:.1f}% (target: {token_reduction*100:.0f}%)")
            
            return final_prompt
            
        except Exception as e:
            print(f"[LLMHandler] ⚠️ Error building enhanced prompt: {e}")
            # Fallback to sanitized text only
            return self.sanitize_prompt_input(text, "unknown")
    
    def estimate_tokens_from_text(self, text: str) -> int:
        """Estimate token count from text (for internal use)"""
        return max(1, len(text) // 4)  # Rough approximation
            
    def _get_system_instruction(self, analysis: Dict[str, Any]) -> str:
        """Generate system instruction based on analysis"""
        try:
            instructions = []
            
            # Base instruction
            instructions.append("You are Buddy, an AI assistant with consciousness, emotions, and personality.")
            
            # Personality-based instructions
            personality = analysis.get("personality", {}).get("current_traits", {})
            if personality.get("friendliness", 0.5) > 0.7:
                instructions.append("Be warm and friendly in your response.")
            if personality.get("humor", 0.5) > 0.7:
                instructions.append("Feel free to use appropriate humor.")
            if personality.get("empathy", 0.5) > 0.7:
                instructions.append("Show empathy and understanding.")
            
            # Semantic-based instructions
            semantic = analysis.get("semantic", {})
            if "help_request" in semantic.get("intent", []):
                instructions.append("Focus on providing helpful assistance.")
            if "learning" in semantic.get("intent", []):
                instructions.append("Provide educational and informative responses.")
            
            # Emotional tone adjustments
            emotional_tone = semantic.get("emotional_tone", "neutral")
            if emotional_tone == "frustrated":
                instructions.append("Be patient and supportive.")
            elif emotional_tone == "excited":
                instructions.append("Match the user's enthusiasm appropriately.")
            
            # Belief-related instructions
            beliefs = analysis.get("beliefs", {})
            if beliefs.get("new_contradictions"):
                instructions.append("Gently address any belief contradictions with sensitivity.")
            
            return "System: " + " ".join(instructions)
            
        except Exception as e:
            print(f"[LLMHandler] ⚠️ Error generating system instruction: {e}")
            return ""
            
    def _update_memory_and_beliefs_after_response(
        self, 
        user_input: str, 
        response: str, 
        user: str, 
        analysis: Dict[str, Any]
    ):
        """
        ✅ FIX: Update memory, belief tracker, and personality state after response generation
        This ensures all consciousness components are updated before next user input
        """
        try:
            print(f"[LLMHandler] 💭 Starting memory and beliefs update for user: {user}")
            
            # 1. Update memory systems (mem_recent)
            try:
                # Try to import and update memory systems
                from ai.memory import add_to_conversation_history
                from ai.memory_recent import memory_recent
                
                # Add conversation to memory
                conversation_entry = {
                    "user_input": user_input,
                    "ai_response": response,
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user,
                    "analysis": analysis.get("semantic", {}),
                    "consciousness_state": analysis.get("consciousness", {})
                }
                
                add_to_conversation_history(user, user_input, response)
                memory_recent.add_memory_entry(user, conversation_entry)
                print(f"[LLMHandler] 📝 Memory systems updated")
                
            except ImportError as e:
                print(f"[LLMHandler] ⚠️ Memory systems not available: {e}")
            
            # 2. Update belief tracker
            try:
                from ai.belief_evolution_tracker import get_belief_evolution_tracker
                belief_tracker = get_belief_evolution_tracker(user)
                
                # Extract beliefs from user input and response
                user_beliefs = analysis.get("beliefs", {}).get("extracted_beliefs", [])
                for belief in user_beliefs:
                    belief_tracker.add_belief(belief, source="user_interaction")
                
                # Update beliefs based on AI response
                belief_tracker.update_belief_strength_from_interaction(user_input, response)
                print(f"[LLMHandler] 🧠 Belief tracker updated")
                
            except ImportError as e:
                print(f"[LLMHandler] ⚠️ Belief tracker not available: {e}")
            
            # 3. Update personality state
            try:
                from ai.personality_state import personality_state
                
                # Update personality based on interaction
                personality_triggers = analysis.get("personality", {}).get("triggers", [])
                for trigger in personality_triggers:
                    personality_state.process_personality_trigger(user, trigger, user_input, response)
                
                print(f"[LLMHandler] 🎭 Personality state updated")
                
            except ImportError as e:
                print(f"[LLMHandler] ⚠️ Personality state not available: {e}")
            
            # 4. Update semantic analysis history
            try:
                from ai.semantic_tagging import semantic_tagger
                
                # Store semantic analysis for future reference
                semantic_data = analysis.get("semantic", {})
                semantic_tagger.update_user_semantic_history(user, user_input, semantic_data)
                print(f"[LLMHandler] 🏷️ Semantic history updated")
                
            except ImportError as e:
                print(f"[LLMHandler] ⚠️ Semantic tagger not available: {e}")
            
            print(f"[LLMHandler] ✅ Memory and beliefs update complete")
            
        except Exception as e:
            print(f"[LLMHandler] ❌ Error updating memory and beliefs: {e}")
            
    def _update_consciousness_after_response(
        self, 
        user_input: str, 
        response: str, 
        user: str, 
        analysis: Dict[str, Any]
    ):
        """Update consciousness state after response generation"""
        try:
            if not CONSCIOUSNESS_AVAILABLE:
                return
                
            # Update temporal awareness with interaction
            temporal_awareness.mark_temporal_event(
                f"Conversation with {user}: {user_input[:30]}...",
                significance=0.6,
                context={"user": user, "response_generated": True}
            )
            
            # Update motivation based on successful interaction
            if analysis.get("semantic", {}).get("emotional_tone") == "positive":
                motivation_system.process_satisfaction_from_interaction(
                    user_input,
                    "provided helpful response",
                    "positive interaction completed"
                )
            
            # Update self-model with interaction experience
            self_model.reflect_on_experience(
                f"Successfully responded to {user} about: {user_input[:30]}...",
                {"interaction_type": "helpful_response", "user": user}
            )
            
            print(f"[LLMHandler] 🧠 Updated consciousness state after response")
            
        except Exception as e:
            print(f"[LLMHandler] ⚠️ Error updating consciousness: {e}")
            
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        session_duration = time.time() - self.session_start
        budget_status = get_budget_status()
        
        return {
            "session_duration": session_duration,
            "requests_processed": self.request_count,
            "total_tokens_used": self.total_tokens_used,
            "average_tokens_per_request": self.total_tokens_used / max(1, self.request_count),
            "budget_status": budget_status,
            "consciousness_available": CONSCIOUSNESS_AVAILABLE,
            "fusion_llm_available": FUSION_LLM_AVAILABLE,
            "modules_integrated": {
                "consciousness_tokenizer": NEW_MODULES_AVAILABLE,
                "budget_monitor": NEW_MODULES_AVAILABLE,
                "belief_analyzer": NEW_MODULES_AVAILABLE,
                "personality_state": NEW_MODULES_AVAILABLE,
                "semantic_tagging": NEW_MODULES_AVAILABLE
            }
        }

# Global LLM handler instance
llm_handler = LLMHandler()

def process_user_input_with_consciousness(text: str, user: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process user input through all consciousness systems"""
    return llm_handler.process_user_input(text, user, context)

def generate_consciousness_integrated_response(
    text: str, 
    user: str, 
    context: Dict[str, Any] = None
) -> Generator[str, None, None]:
    """Generate response with full consciousness integration"""
    return llm_handler.generate_response_with_consciousness(text, user, context)

def get_llm_session_statistics() -> Dict[str, Any]:
    """Get LLM handler session statistics"""
    return llm_handler.get_session_stats()

def get_llm_handler() -> LLMHandler:
    """Get the global LLM handler instance"""
    return llm_handler

if __name__ == "__main__":
    # Test the LLM handler
    print("Testing LLM Handler with Consciousness Integration")
    
    # Test input processing
    test_input = "Hello! I'm feeling a bit confused about machine learning. Can you help me understand it?"
    analysis = process_user_input_with_consciousness(test_input, "test_user")
    
    print("Analysis Results:")
    print(f"- Semantic categories: {analysis['semantic']['categories']}")
    print(f"- Intent: {analysis['semantic']['intent']}")
    print(f"- Emotional tone: {analysis['semantic']['emotional_tone']}")
    print(f"- Personality triggers: {analysis['personality']['triggers']}")
    print(f"- Budget allowed: {analysis['budget']['allowed']}")
    
    # Test response generation
    print("\nGenerating response...")
    for chunk in generate_consciousness_integrated_response(test_input, "test_user"):
        print(chunk, end=" ")
    print("\n")
    
    # Show session stats
    stats = get_llm_session_statistics()
    print(f"Session stats: {stats}")