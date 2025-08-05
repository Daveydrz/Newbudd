"""
Unified Consciousness Context Injector

This module creates a batched consciousness injection system that eliminates 
the critical issue of multiple separate LLM calls for consciousness traits,
emotional state, memory, and internal thoughts.

PROBLEM SOLVED:
- Multiple separate calls to personality_injector, emotional_state_injector, 
  belief_tracker, temporal_awareness, long_term_memory, thematic_belief_fusion
- Each making separate LLM calls causing latency and TTS confusion
- Prompt bloat from multiple injections

SOLUTION:
- Single unified function build_consciousness_context(user_id) 
- Batches all consciousness data into one operation
- Caches results for 30 seconds to prevent duplicate calls
- Returns compressed consciousness context for system message injection
"""

import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class UnifiedConsciousnessContext:
    """Unified consciousness context for LLM injection"""
    timestamp: str
    user_id: str
    
    # Compressed consciousness summary
    consciousness_summary: str
    
    # Individual components (for debugging)
    memory_context: str
    personality_traits: str  
    emotional_state: str
    temporal_context: str
    goals_context: str
    beliefs_context: str
    
    # Metadata
    cache_key: str
    token_count: int
    compression_level: str

class UnifiedContextInjector:
    """Unified consciousness context injector with smart caching"""
    
    def __init__(self):
        self.context_cache = {}
        self.cache_duration = 30  # 30 seconds cache
        self.max_tokens = 150  # Maximum tokens for consciousness context
        
        # Component availability
        self._check_component_availability()
        
        print("[ContextInjector] 🧠 Unified consciousness context injector initialized")
    
    def _check_component_availability(self):
        """Check which consciousness components are available"""
        self.components_available = {
            'memory': False,
            'personality': False, 
            'emotions': False,
            'temporal': False,
            'goals': False,
            'beliefs': False,
            'mood': False
        }
        
        # Check memory system
        try:
            from ai.memory import get_user_memory
            self.components_available['memory'] = True
        except ImportError:
            pass
            
        # Check personality system
        try:
            from ai.personality_profile import get_personality_profile
            self.components_available['personality'] = True
        except ImportError:
            pass
            
        # Check emotion system
        try:
            from ai.emotion import get_current_emotional_state
            self.components_available['emotions'] = True
        except ImportError:
            pass
            
        # Check temporal awareness
        try:
            from ai.temporal_awareness import temporal_awareness
            self.components_available['temporal'] = True
        except ImportError:
            pass
            
        # Check goals system
        try:
            from ai.goal_manager import get_user_goal_manager
            self.components_available['goals'] = True
        except ImportError:
            pass
            
        # Check beliefs system
        try:
            from ai.belief_evolution_tracker import get_belief_tracker
            self.components_available['beliefs'] = True
        except ImportError:
            pass
            
        # Check mood system
        try:
            from ai.mood_manager import get_mood_manager
            self.components_available['mood'] = True
        except ImportError:
            pass
        
        available_count = sum(self.components_available.values())
        print(f"[ContextInjector] ✅ {available_count}/7 consciousness components available")
    
    def build_consciousness_context(self, user_id: str, use_cache: bool = True) -> UnifiedConsciousnessContext:
        """
        Build unified consciousness context by batching ALL consciousness data
        
        This replaces separate calls to:
        - personality_injector
        - emotional_state_injector  
        - belief_tracker
        - temporal_awareness
        - long_term_memory
        - thematic_belief_fusion
        
        Returns compressed consciousness context for system message injection
        """
        try:
            # Create cache key
            cache_key = f"consciousness_{user_id}_{int(time.time() // self.cache_duration)}"
            
            # Check cache first
            if use_cache and cache_key in self.context_cache:
                cached_context = self.context_cache[cache_key]
                print(f"[ContextInjector] 📋 Using cached consciousness context for {user_id}")
                return cached_context
            
            print(f"[ContextInjector] 🧠 Building unified consciousness context for {user_id}")
            
            # Gather all consciousness data in batched operations
            consciousness_data = self._gather_all_consciousness_data(user_id)
            
            # Create unified context
            unified_context = self._create_unified_context(user_id, consciousness_data, cache_key)
            
            # Cache the result
            if use_cache:
                self.context_cache[cache_key] = unified_context
                
                # Clean old cache entries
                self._clean_old_cache_entries()
            
            print(f"[ContextInjector] ✅ Unified consciousness context built: {unified_context.token_count} tokens")
            return unified_context
            
        except Exception as e:
            print(f"[ContextInjector] ❌ Error building consciousness context: {e}")
            return self._create_fallback_context(user_id)
    
    def _gather_all_consciousness_data(self, user_id: str) -> Dict[str, Any]:
        """Gather ALL consciousness data in batched operations"""
        consciousness_data = {
            'memory': {},
            'personality': {},
            'emotions': {},
            'temporal': {},
            'goals': {},
            'beliefs': {},
            'mood': {}
        }
        
        # Batch gather memory context
        if self.components_available['memory']:
            try:
                from ai.memory import get_user_memory
                user_memory = get_user_memory(user_id)
                
                # Get memory summary (not full extraction - just context)
                memory_summary = user_memory.get_memory_context()
                recent_memories = user_memory.get_recent_interactions(limit=3)
                
                consciousness_data['memory'] = {
                    'summary': memory_summary[:100] if memory_summary else "No memories",
                    'recent': [str(m)[:50] for m in recent_memories] if recent_memories else [],
                    'available': True
                }
                print(f"[ContextInjector] 💭 Memory context gathered")
                
            except Exception as e:
                print(f"[ContextInjector] ⚠️ Memory gathering error: {e}")
                consciousness_data['memory'] = {'available': False}
        
        # Batch gather personality traits
        if self.components_available['personality']:
            try:
                from ai.personality_profile import get_personality_profile
                personality = get_personality_profile(user_id)
                
                consciousness_data['personality'] = {
                    'traits': personality.get('primary_traits', {}) if personality else {},
                    'style': personality.get('interaction_style', 'balanced') if personality else 'balanced',
                    'available': True
                }
                print(f"[ContextInjector] 🎭 Personality traits gathered")
                
            except Exception as e:
                print(f"[ContextInjector] ⚠️ Personality gathering error: {e}")
                consciousness_data['personality'] = {'available': False}
        
        # Batch gather emotional state  
        if self.components_available['emotions']:
            try:
                from ai.emotion import get_current_emotional_state
                emotion_state = get_current_emotional_state()
                
                consciousness_data['emotions'] = {
                    'primary': emotion_state.get('current_emotion', 'neutral'),
                    'intensity': emotion_state.get('intensity', 0.5),
                    'valence': emotion_state.get('valence', 0.0),
                    'available': True
                }
                print(f"[ContextInjector] 😊 Emotional state gathered")
                
            except Exception as e:
                print(f"[ContextInjector] ⚠️ Emotion gathering error: {e}")
                consciousness_data['emotions'] = {'available': False}
        
        # Batch gather temporal awareness
        if self.components_available['temporal']:
            try:
                from ai.temporal_awareness import temporal_awareness
                temporal_context = temporal_awareness.get_current_time_context()
                
                consciousness_data['temporal'] = {
                    'time_of_day': temporal_context.get('time_of_day', 'unknown'),
                    'context': temporal_context.get('context', 'present'),
                    'awareness_level': temporal_context.get('awareness_level', 0.5),
                    'available': True
                }
                print(f"[ContextInjector] ⏰ Temporal awareness gathered")
                
            except Exception as e:
                print(f"[ContextInjector] ⚠️ Temporal gathering error: {e}")
                consciousness_data['temporal'] = {'available': False}
        
        # Batch gather goals context
        if self.components_available['goals']:
            try:
                from ai.goal_manager import get_user_goal_manager
                goal_manager = get_user_goal_manager(user_id)
                active_goals = goal_manager.get_active_goals() if goal_manager else []
                
                consciousness_data['goals'] = {
                    'active_count': len(active_goals),
                    'primary_goal': active_goals[0].description[:50] if active_goals else "No active goals",
                    'available': True
                }
                print(f"[ContextInjector] 🎯 Goals context gathered")
                
            except Exception as e:
                print(f"[ContextInjector] ⚠️ Goals gathering error: {e}")
                consciousness_data['goals'] = {'available': False}
        
        # Batch gather beliefs context
        if self.components_available['beliefs']:
            try:
                from ai.belief_evolution_tracker import get_belief_tracker
                belief_tracker = get_belief_tracker(user_id)
                active_beliefs = belief_tracker.get_active_beliefs() if belief_tracker else []
                
                consciousness_data['beliefs'] = {
                    'count': len(active_beliefs),
                    'primary': active_beliefs[0].content[:50] if active_beliefs else "Core helpful values",
                    'available': True
                }
                print(f"[ContextInjector] 🧠 Beliefs context gathered")
                
            except Exception as e:
                print(f"[ContextInjector] ⚠️ Beliefs gathering error: {e}")
                consciousness_data['beliefs'] = {'available': False}
        
        # Batch gather mood context
        if self.components_available['mood']:
            try:
                from ai.mood_manager import get_mood_manager
                mood_manager = get_mood_manager(user_id)
                mood_state = mood_manager.get_current_mood() if mood_manager else None
                
                consciousness_data['mood'] = {
                    'current': mood_state.value if mood_state else 'neutral',
                    'stability': 0.7,  # Default
                    'available': True
                }
                print(f"[ContextInjector] 🌈 Mood context gathered")
                
            except Exception as e:
                print(f"[ContextInjector] ⚠️ Mood gathering error: {e}")
                consciousness_data['mood'] = {'available': False}
        
        return consciousness_data
    
    def _create_unified_context(self, user_id: str, consciousness_data: Dict[str, Any], cache_key: str) -> UnifiedConsciousnessContext:
        """Create unified consciousness context from gathered data"""
        
        # Build individual component contexts
        memory_context = self._build_memory_context(consciousness_data['memory'])
        personality_context = self._build_personality_context(consciousness_data['personality']) 
        emotional_context = self._build_emotional_context(consciousness_data['emotions'])
        temporal_context = self._build_temporal_context(consciousness_data['temporal'])
        goals_context = self._build_goals_context(consciousness_data['goals'])
        beliefs_context = self._build_beliefs_context(consciousness_data['beliefs'])
        
        # Create unified consciousness summary
        consciousness_parts = []
        
        # Add emotion and mood (most important for response tone)
        if consciousness_data['emotions'].get('available', False):
            emotion = consciousness_data['emotions']['primary']
            intensity = consciousness_data['emotions']['intensity']
            consciousness_parts.append(f"Emotion:{emotion}({intensity:.1f})")
        
        # Add personality style
        if consciousness_data['personality'].get('available', False):
            style = consciousness_data['personality']['style']
            consciousness_parts.append(f"Style:{style}")
        
        # Add temporal context
        if consciousness_data['temporal'].get('available', False):
            time_context = consciousness_data['temporal']['time_of_day']
            consciousness_parts.append(f"Time:{time_context}")
        
        # Add goals if active
        if consciousness_data['goals'].get('available', False) and consciousness_data['goals']['active_count'] > 0:
            goal_count = consciousness_data['goals']['active_count']
            consciousness_parts.append(f"Goals:{goal_count}active")
        
        # Add memory context if available
        if consciousness_data['memory'].get('available', False) and consciousness_data['memory'].get('recent'):
            memory_indicator = f"Memory:{len(consciousness_data['memory']['recent'])}recent"
            consciousness_parts.append(memory_indicator)
        
        # Create compressed consciousness summary
        consciousness_summary = "[" + "|".join(consciousness_parts) + "]"
        
        # Estimate token count
        token_count = len(consciousness_summary.split())
        
        # Determine compression level
        if token_count <= 30:
            compression_level = "optimal"
        elif token_count <= 50:
            compression_level = "moderate" 
        else:
            compression_level = "high"
            # Apply aggressive compression if too long
            consciousness_summary = consciousness_summary[:200] + "...]"
            token_count = len(consciousness_summary.split())
        
        return UnifiedConsciousnessContext(
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            consciousness_summary=consciousness_summary,
            memory_context=memory_context,
            personality_traits=personality_context,
            emotional_state=emotional_context,
            temporal_context=temporal_context,
            goals_context=goals_context,
            beliefs_context=beliefs_context,
            cache_key=cache_key,
            token_count=token_count,
            compression_level=compression_level
        )
    
    def _build_memory_context(self, memory_data: Dict[str, Any]) -> str:
        """Build memory context string"""
        if not memory_data.get('available', False):
            return "Memory:unavailable"
        
        summary = memory_data.get('summary', '')
        recent = memory_data.get('recent', [])
        
        if recent:
            return f"Memory:{len(recent)}recent"
        elif summary:
            return f"Memory:context_available"
        else:
            return "Memory:none"
    
    def _build_personality_context(self, personality_data: Dict[str, Any]) -> str:
        """Build personality context string"""
        if not personality_data.get('available', False):
            return "Personality:balanced"
        
        style = personality_data.get('style', 'balanced')
        traits = personality_data.get('traits', {})
        
        if traits:
            dominant_trait = max(traits.items(), key=lambda x: x[1])[0] if traits else 'balanced'
            return f"Personality:{style}({dominant_trait})"
        else:
            return f"Personality:{style}"
    
    def _build_emotional_context(self, emotion_data: Dict[str, Any]) -> str:
        """Build emotional context string"""
        if not emotion_data.get('available', False):
            return "Emotion:neutral"
        
        primary = emotion_data.get('primary', 'neutral')
        intensity = emotion_data.get('intensity', 0.5)
        
        return f"Emotion:{primary}({intensity:.1f})"
    
    def _build_temporal_context(self, temporal_data: Dict[str, Any]) -> str:
        """Build temporal context string"""
        if not temporal_data.get('available', False):
            return "Time:present"
        
        time_of_day = temporal_data.get('time_of_day', 'unknown')
        context = temporal_data.get('context', 'present')
        
        return f"Time:{time_of_day}({context})"
    
    def _build_goals_context(self, goals_data: Dict[str, Any]) -> str:
        """Build goals context string"""
        if not goals_data.get('available', False):
            return "Goals:none"
        
        count = goals_data.get('active_count', 0)
        if count > 0:
            return f"Goals:{count}active"
        else:
            return "Goals:none"
    
    def _build_beliefs_context(self, beliefs_data: Dict[str, Any]) -> str:
        """Build beliefs context string"""
        if not beliefs_data.get('available', False):
            return "Beliefs:core_values"
        
        count = beliefs_data.get('count', 0)
        if count > 0:
            return f"Beliefs:{count}active"
        else:
            return "Beliefs:core_values"
    
    def _create_fallback_context(self, user_id: str) -> UnifiedConsciousnessContext:
        """Create fallback consciousness context when gathering fails"""
        fallback_summary = "[Emotion:neutral|Style:balanced|Time:present|Memory:basic]"
        
        return UnifiedConsciousnessContext(
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            consciousness_summary=fallback_summary,
            memory_context="Memory:basic",
            personality_traits="Personality:balanced",
            emotional_state="Emotion:neutral",
            temporal_context="Time:present",
            goals_context="Goals:none",
            beliefs_context="Beliefs:core_values",
            cache_key=f"fallback_{user_id}_{int(time.time())}",
            token_count=len(fallback_summary.split()),
            compression_level="minimal"
        )
    
    def _clean_old_cache_entries(self):
        """Clean old cache entries to prevent memory bloat"""
        current_time = time.time()
        expired_keys = []
        
        for key, context in self.context_cache.items():
            # Parse timestamp from cache key
            try:
                key_time = int(key.split('_')[-1]) * self.cache_duration
                if current_time - key_time > self.cache_duration * 2:  # Double the cache duration
                    expired_keys.append(key)
            except (ValueError, IndexError):
                # Remove malformed keys
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.context_cache[key]
        
        if expired_keys:
            print(f"[ContextInjector] 🧹 Cleaned {len(expired_keys)} expired cache entries")
    
    def get_context_for_system_message(self, user_id: str) -> str:
        """Get consciousness context formatted for system message injection"""
        try:
            context = self.build_consciousness_context(user_id)
            
            # Format for system message injection
            system_context = f"Buddy's current consciousness state: {context.consciousness_summary}"
            
            print(f"[ContextInjector] 📤 System message context: {len(system_context)} chars")
            return system_context
            
        except Exception as e:
            print(f"[ContextInjector] ❌ Error getting system context: {e}")
            return "Buddy's current consciousness state: [Emotion:neutral|Style:balanced|Time:present]"
    
    def clear_cache(self):
        """Clear consciousness context cache"""
        self.context_cache.clear()
        print("[ContextInjector] 🗑️ Consciousness context cache cleared")

# Global instance
context_injector = UnifiedContextInjector()

def build_consciousness_context(user_id: str, use_cache: bool = True) -> UnifiedConsciousnessContext:
    """
    Build unified consciousness context - MAIN API FUNCTION
    
    This replaces all separate injector calls:
    - personality_injector
    - emotional_state_injector
    - belief_tracker
    - temporal_awareness  
    - long_term_memory
    - thematic_belief_fusion
    
    Returns batched consciousness context for system message injection
    """
    return context_injector.build_consciousness_context(user_id, use_cache)

def get_consciousness_for_system_message(user_id: str) -> str:
    """Get consciousness context formatted for LLM system message injection"""
    return context_injector.get_context_for_system_message(user_id)

def clear_consciousness_cache():
    """Clear consciousness context cache (for testing/debugging)"""
    context_injector.clear_cache()