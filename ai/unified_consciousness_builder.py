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
            from ai.personality_profile import get_personality_profile_manager
            profile_manager = get_personality_profile_manager(username)
            # Fix method call
            if hasattr(profile_manager, 'get_profile'):
                profile = profile_manager.get_profile()
            elif hasattr(profile_manager, 'personality_config'):
                profile = profile_manager.personality_config
            else:
                profile = {}
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
            # Try to get belief state from existing belief system
            from ai.belief_analyzer import BeliefAnalyzer
            belief_analyzer = BeliefAnalyzer()
            # Fix method call
            if hasattr(belief_analyzer, 'get_summary'):
                belief_summary = belief_analyzer.get_summary()
            elif hasattr(belief_analyzer, 'beliefs'):
                belief_summary = {
                    'core_beliefs': list(belief_analyzer.beliefs.keys())[:3] if belief_analyzer.beliefs else [],
                    'recent_beliefs': [],
                    'contradictions': []
                }
            else:
                belief_summary = {'core_beliefs': [], 'recent_beliefs': [], 'contradictions': []}
            
            return {
                'core_beliefs': belief_summary.get('core_beliefs', [])[:3],  # Top 3 core beliefs
                'recent_beliefs': belief_summary.get('recent_beliefs', [])[:2],  # 2 recent beliefs
                'contradictions': belief_summary.get('contradictions', [])[:1]  # 1 contradiction
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
            # Use the correct method name for getting recent memories
            if hasattr(user_memory, 'get_recent_conversation_history'):
                recent_memories = user_memory.get_recent_conversation_history(limit=3)
            elif hasattr(user_memory, 'memories'):
                recent_memories = user_memory.memories[-3:] if user_memory.memories else []
            else:
                recent_memories = []
            
            return {
                'recent_memories': [{'topic': str(m).get('topic', str(m)[:50]), 'significance': 0.5} 
                                  for m in recent_memories],
                'memory_count': len(user_memory.memories) if hasattr(user_memory, 'memories') else 0
            }
        except Exception as e:
            print(f"[UnifiedConsciousness] ⚠️ Memory state error: {e}")
            return {'recent_memories': [], 'memory_count': 0}
    
    def _gather_goals_state(self, username: str) -> Dict[str, Any]:
        """Gather goal management data"""
        try:
            # Try to get goal state from goal engine
            from ai.goal_engine import GoalEngine
            goal_engine = GoalEngine(username)
            if hasattr(goal_engine, 'get_active_goals'):
                active_goals = goal_engine.get_active_goals()[:2]
            elif hasattr(goal_engine, 'goals'):
                active_goals = list(goal_engine.goals.values())[:2] if goal_engine.goals else []
            else:
                active_goals = []
            return {
                'active_goals': [{'description': str(g)[:50], 'priority': 0.5} 
                               for g in active_goals],
                'goal_progress': 0.5 if active_goals else 0
            }
        except Exception as e:
            print(f"[UnifiedConsciousness] ⚠️ Goals state error: {e}")
            return {'active_goals': [], 'goal_progress': 0}
    
    def _compress_consciousness_context(self, context_data: Dict[str, Any], username: str) -> str:
        """Compress consciousness data into compact context string"""
        try:
            # ✅ FIX: Use simple compression instead of LLM call to prevent TTS interference
            emotion = context_data['emotional']['primary_emotion']
            memory_count = context_data['memory']['memory_count']
            goal_count = len(context_data['goals']['active_goals'])
            belief_count = len(context_data['belief']['core_beliefs'])
            
            # Create simple compressed context without LLM call
            compressed = f"[CONSCIOUSNESS] User: {username}, Emotion: {emotion}, {memory_count} memories, {goal_count} goals, {belief_count} beliefs"
            
            return compressed.strip()[:200]  # Max 200 chars
            
        except Exception as e:
            print(f"[UnifiedConsciousness] ⚠️ Compression error: {e}")
            # Simple fallback compression
            emotion = context_data.get('emotional', {}).get('primary_emotion', 'neutral')
            memory_count = context_data.get('memory', {}).get('memory_count', 0)
            goal_count = len(context_data.get('goals', {}).get('active_goals', []))
            return f"[CONSCIOUSNESS] User: {username}, Emotion: {emotion}, {memory_count} memories, {goal_count} goals"
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