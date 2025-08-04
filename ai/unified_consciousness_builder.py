# ai/unified_consciousness_builder.py - Batched consciousness context injection system
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass 
class ConsciousnessContext:
    """Consolidated consciousness context data"""
    personality_state: Dict[str, Any]
    emotional_state: Dict[str, Any] 
    belief_state: Dict[str, Any]
    temporal_state: Dict[str, Any]
    memory_state: Dict[str, Any]
    goals_state: Dict[str, Any]
    compressed_context: str
    token_count: int

class UnifiedConsciousnessBuilder:
    """Builds consciousness context in a single batch operation"""
    
    def __init__(self):
        self.cache_timeout = 30  # 30 seconds cache for consciousness state
        self.last_context: Optional[ConsciousnessContext] = None
        self.last_context_time = 0
        
    def _gather_personality_state(self, username: str) -> Dict[str, Any]:
        """Gather personality injector data"""
        try:
            from ai.personality_profile import get_personality_profile
            profile = get_personality_profile(username)
            return {
                'traits': profile.get('traits', {}),
                'communication_style': profile.get('communication_style', 'conversational'),
                'preferences': profile.get('preferences', {})
            }
        except Exception as e:
            print(f"[UnifiedConsciousness] ⚠️ Personality state error: {e}")
            return {'traits': {}, 'communication_style': 'conversational', 'preferences': {}}
    
    def _gather_emotional_state(self, username: str) -> Dict[str, Any]:
        """Gather emotional state injector data"""
        try:
            from ai.emotion import get_current_emotional_state
            emotion_state = get_current_emotional_state()
            return {
                'primary_emotion': emotion_state.get('current_emotion', 'neutral'),
                'intensity': emotion_state.get('intensity', 0.5),
                'emotional_context': emotion_state.get('context', '')
            }
        except Exception as e:
            print(f"[UnifiedConsciousness] ⚠️ Emotional state error: {e}")
            return {'primary_emotion': 'neutral', 'intensity': 0.5, 'emotional_context': ''}
    
    def _gather_belief_state(self, username: str) -> Dict[str, Any]:
        """Gather belief tracker data"""
        try:
            from ai.belief_evolution_tracker import get_belief_tracker
            belief_tracker = get_belief_tracker(username)
            active_beliefs = belief_tracker.get_active_beliefs()
            return {
                'core_beliefs': active_beliefs.get('core', [])[:3],  # Top 3 core beliefs
                'recent_beliefs': active_beliefs.get('recent', [])[:2],  # 2 recent beliefs
                'contradictions': belief_tracker.detect_contradictions()[:1]  # 1 contradiction
            }
        except Exception as e:
            print(f"[UnifiedConsciousness] ⚠️ Belief state error: {e}")
            return {'core_beliefs': [], 'recent_beliefs': [], 'contradictions': []}
    
    def _gather_temporal_state(self, username: str) -> Dict[str, Any]:
        """Gather temporal awareness data"""
        try:
            from ai.temporal_awareness import temporal_awareness
            temporal_context = temporal_awareness.get_current_time_context()
            return {
                'current_timeframe': temporal_context.get('current_timeframe', 'present'),
                'recent_events': temporal_context.get('recent_events', [])[:2],  # 2 recent events
                'temporal_patterns': temporal_context.get('patterns', [])[:1]  # 1 pattern
            }
        except Exception as e:
            print(f"[UnifiedConsciousness] ⚠️ Temporal state error: {e}")
            return {'current_timeframe': 'present', 'recent_events': [], 'temporal_patterns': []}
    
    def _gather_memory_state(self, username: str) -> Dict[str, Any]:
        """Gather long-term memory data"""
        try:
            from ai.memory import get_user_memory
            user_memory = get_user_memory(username)
            recent_memories = user_memory.get_recent_memories(limit=3)
            return {
                'recent_memories': [{'topic': m.get('topic', ''), 'significance': m.get('significance', 0.5)} 
                                  for m in recent_memories],
                'memory_count': len(user_memory.memories) if hasattr(user_memory, 'memories') else 0
            }
        except Exception as e:
            print(f"[UnifiedConsciousness] ⚠️ Memory state error: {e}")
            return {'recent_memories': [], 'memory_count': 0}
    
    def _gather_goals_state(self, username: str) -> Dict[str, Any]:
        """Gather goal management data"""
        try:
            from ai.goal_manager import get_user_goal_manager
            goal_manager = get_user_goal_manager(username)
            active_goals = goal_manager.get_active_goals()[:2]  # Top 2 active goals
            return {
                'active_goals': [{'description': g.description[:50], 'priority': g.priority} 
                               for g in active_goals],
                'goal_progress': sum(g.progress for g in active_goals) / len(active_goals) if active_goals else 0
            }
        except Exception as e:
            print(f"[UnifiedConsciousness] ⚠️ Goals state error: {e}")
            return {'active_goals': [], 'goal_progress': 0}
    
    def _compress_consciousness_context(self, context_data: Dict[str, Any], username: str) -> str:
        """Compress consciousness data into compact context string"""
        try:
            from ai.llm_handler import LLMHandler
            
            # Create compression prompt
            compression_prompt = f"""Compress this consciousness data into a concise context (max 100 tokens):

Personality: {context_data['personality']}
Emotion: {context_data['emotional']['primary_emotion']} (intensity: {context_data['emotional']['intensity']})
Beliefs: {len(context_data['belief']['core_beliefs'])} core beliefs, {len(context_data['belief']['contradictions'])} contradictions
Memory: {context_data['memory']['memory_count']} memories, recent: {[m['topic'] for m in context_data['memory']['recent_memories']]}
Goals: {len(context_data['goals']['active_goals'])} active goals, progress: {context_data['goals']['goal_progress']:.1f}
Temporal: {context_data['temporal']['current_timeframe']}, {len(context_data['temporal']['recent_events'])} recent events

Create a compressed consciousness context for {username} that captures the essential state for response generation."""

            llm_handler = LLMHandler()
            compressed = llm_handler.generate_response_with_consciousness(
                compression_prompt, username, {"context": "consciousness_compression"}
            )
            
            # Fallback if compression fails
            if not compressed or len(compressed.strip()) < 10:
                compressed = f"User: {username}, Emotion: {context_data['emotional']['primary_emotion']}, {context_data['memory']['memory_count']} memories"
            
            return compressed.strip()[:200]  # Max 200 chars
            
        except Exception as e:
            print(f"[UnifiedConsciousness] ⚠️ Compression error: {e}")
            # Simple fallback compression
            emotion = context_data['emotional']['primary_emotion']
            memory_count = context_data['memory']['memory_count']
            goal_count = len(context_data['goals']['active_goals'])
            return f"User: {username}, Emotion: {emotion}, {memory_count} memories, {goal_count} goals"
    
    def build_consciousness_context(self, username: str, force_refresh: bool = False) -> ConsciousnessContext:
        """Build unified consciousness context with caching"""
        
        # Check cache first
        if not force_refresh and self.last_context and (time.time() - self.last_context_time) < self.cache_timeout:
            print(f"[UnifiedConsciousness] ♻️ Using cached consciousness context for {username}")
            return self.last_context
        
        print(f"[UnifiedConsciousness] 🧠 Building fresh consciousness context for {username}")
        
        # Gather all consciousness data in parallel
        context_data = {
            'personality': self._gather_personality_state(username),
            'emotional': self._gather_emotional_state(username),
            'belief': self._gather_belief_state(username),
            'temporal': self._gather_temporal_state(username),
            'memory': self._gather_memory_state(username),
            'goals': self._gather_goals_state(username)
        }
        
        # Compress into context string
        compressed_context = self._compress_consciousness_context(context_data, username)
        
        # Calculate token count (rough estimate)
        token_count = len(compressed_context.split()) + sum(
            len(str(data).split()) for data in context_data.values()
        )
        
        # Create consciousness context
        consciousness_context = ConsciousnessContext(
            personality_state=context_data['personality'],
            emotional_state=context_data['emotional'],
            belief_state=context_data['belief'],
            temporal_state=context_data['temporal'],
            memory_state=context_data['memory'],
            goals_state=context_data['goals'],
            compressed_context=compressed_context,
            token_count=token_count
        )
        
        # Cache the result
        self.last_context = consciousness_context
        self.last_context_time = time.time()
        
        print(f"[UnifiedConsciousness] ✅ Consciousness context built ({token_count} tokens)")
        return consciousness_context
    
    def get_consciousness_injection_string(self, username: str) -> str:
        """Get consciousness context as injection string for prompt"""
        context = self.build_consciousness_context(username)
        return f"[CONSCIOUSNESS_CONTEXT] {context.compressed_context}"

# Global unified consciousness builder
unified_consciousness_builder = UnifiedConsciousnessBuilder()

def get_unified_consciousness_context(username: str, force_refresh: bool = False) -> ConsciousnessContext:
    """Get unified consciousness context (main entry point)"""
    return unified_consciousness_builder.build_consciousness_context(username, force_refresh)

def get_consciousness_injection(username: str) -> str:
    """Get consciousness injection string for prompt"""
    return unified_consciousness_builder.get_consciousness_injection_string(username)