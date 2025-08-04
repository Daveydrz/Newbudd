# ai/human_memory_smart.py - Smart LLM-based life event detection
import json
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import re
from ai.memory import get_user_memory, add_to_conversation_history
from ai.chat import ask_kobold  # Use your existing LLM connection

class SmartHumanLikeMemory:
    """🧠 Smart human-like memory using LLM for event detection"""
    
    def __init__(self, username: str):
        self.username = username
        self.memory_dir = f"memory/{username}"
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Get the existing MEGA-INTELLIGENT memory system
        self.mega_memory = get_user_memory(username)
        
        # Smart memory storage
        self.appointments = self.load_memory('smart_appointments.json')
        self.life_events = self.load_memory('smart_life_events.json') 
        self.conversation_highlights = self.load_memory('smart_highlights.json')
        
        # Session tracking
        self.context_used_this_session = set()
        
        print(f"[SmartMemory] 🧠 Smart LLM-based memory initialized for {username}")
    
    def extract_and_store_human_memories(self, text: str):
        """🎯 Smart LLM-based memory extraction with BULLETPROOF filtering"""
        
        # Also use the existing MEGA-INTELLIGENT extraction
        self.mega_memory.extract_memories_from_text(text)
        
        # Use LLM to intelligently detect events (only if passes all filters)
        detected_events = self._smart_detect_events(text)
        
        # Store detected events
        for event in detected_events:
            if event['type'] == 'appointment':
                self.appointments.append(event)
                print(f"[SmartMemory] 📅 Smart appointment: {event['topic']} on {event['date']}")
                # CRITICAL: Also add to regular memory system for retrieval
                self._add_to_regular_memory(event)
                
            elif event['type'] == 'life_event':
                self.life_events.append(event)
                print(f"[SmartMemory] 📝 Smart life event: {event['topic']} on {event['date']} ({event['emotion']})")
                # CRITICAL: Also add to regular memory system for retrieval
                self._add_to_regular_memory(event)
                
            elif event['type'] == 'highlight':
                self.conversation_highlights.append(event)
                print(f"[SmartMemory] 💬 Smart highlight: {event['topic']}")
        
        # Save memories
        if detected_events:  # Only save if we actually found events
            self.save_memory(self.appointments, 'smart_appointments.json')
            self.save_memory(self.life_events, 'smart_life_events.json')
            self.save_memory(self.conversation_highlights, 'smart_highlights.json')
    
    def _add_to_regular_memory(self, event: Dict):
        """Add detected event to regular memory system for retrieval"""
        try:
            # Create a memory fact from the event
            fact_key = event['topic'].replace(' ', '_').lower()
            fact_value = self._create_readable_memory_value(event)
            
            # Add to personal facts with correct PersonalFact constructor
            from ai.memory import PersonalFact
            fact = PersonalFact(
                key=fact_key,
                value=fact_value,
                date_learned=event.get('date', event.get('created', '')),
                confidence=0.9
            )
            
            # Set additional attributes after construction
            fact.emotional_significance = 0.7 if event.get('emotion') in ['happy', 'excited'] else 0.3
            fact.source_context = event.get('original_text', '')
            
            self.mega_memory.personal_facts[fact_key] = fact
            print(f"[SmartMemory] ➕ Added to regular memory: {fact_key} = {fact_value}")
            
        except Exception as e:
            print(f"[SmartMemory] ⚠️ Error adding to regular memory: {e}")
            # Fallback: add directly to working memory
            try:
                fact_key = event['topic'].replace(' ', '_').lower()
                fact_value = self._create_readable_memory_value(event)
                self.mega_memory.working_memory.last_action = f"{fact_key}: {fact_value}"
                print(f"[SmartMemory] 🔄 Added to working memory instead: {fact_key}")
            except Exception as e2:
                print(f"[SmartMemory] ❌ Fallback also failed: {e2}")
    
    def _create_readable_memory_value(self, event: Dict) -> str:
        """Create a readable memory value from an event"""
        topic = event.get('topic', '').replace('_', ' ')
        date = event.get('date', '')
        emotion = event.get('emotion', '')
        original_text = event.get('original_text', '')
        
        # Extract key information based on event type
        if event.get('type') == 'appointment':
            return f"{topic} scheduled for {date}"
        elif event.get('type') == 'life_event':
            # For life events, try to extract more context
            if 'mcdonalds' in topic.lower() or 'mcdonald' in topic.lower():
                if 'with_' in topic:
                    companion = topic.split('with_')[-1].replace('_', ' ')
                    return f"mcdonalds with {companion}"
                return "mcdonalds"
            elif 'birthday' in topic.lower():
                if '_' in topic:
                    person = topic.split('_')[0]
                    return f"{person} birthday on {date}"
                return f"birthday on {date}"
            elif 'met_' in topic:
                parts = topic.replace('met_', '').split('_at_')
                if len(parts) == 2:
                    person, place = parts
                    return f"met {person.replace('_', ' ')} at {place.replace('_', ' ')}"
            return topic.replace('_', ' ')
        else:
            return topic.replace('_', ' ')
    
    def _is_casual_conversation(self, text: str) -> bool:
        """🛡️ BULLETPROOF filter to block casual conversation from LLM"""
        text_lower = text.lower().strip()
        
        # Block ALL questions TO Buddy
        question_to_buddy_patterns = [
            r'how.+are.+you',           # "How are you today?"
            r'how.+you.+doing',         # "How you doing?"
            r'what.+about.+you',        # "What about you?"
            r'how.+your.+day',          # "How was your day?"
            r'what.+your.+plan',        # "What's your plan?"
            r'what.+you.+think',        # "What do you think?"
            r'do.+you.+know',           # "Do you know..."
            r'can.+you.+help',          # "Can you help me?"
            r'can.+you.+tell',          # "Can you tell me?"
            r'what.+is.+your',          # "What is your..."
            r'where.+are.+you',         # "Where are you?"
            r'what.+time.+is.+it',      # "What time is it?"
            r'tell.+me.+about',         # "Tell me about..."
            r'explain.+to.+me',         # "Explain to me..."
            r'how.+does.+this',         # "How does this work?"
            r'why.+is.+this',           # "Why is this..."
            r'what.+does.+this',        # "What does this mean?"
        ]
        
        # Block ALL greetings and pleasantries
        greeting_patterns = [
            r'^thanks?\s+buddy',        # "Thanks buddy"
            r'^thank\s+you',            # "Thank you"
            r'^hello\b',                # "Hello"
            r'^hi\b',                   # "Hi"
            r'^hey\b',                  # "Hey"
            r'^good\s+morning',         # "Good morning"
            r'^good\s+afternoon',       # "Good afternoon" 
            r'^good\s+evening',         # "Good evening"
            r'^good\s+night',           # "Good night"
            r'nice.+talking',           # "Nice talking to you"
            r'see.+you.+later',         # "See you later"
            r'goodbye',                 # "Goodbye"
            r'bye',                     # "Bye"
            r'talk.+to.+you.+later',    # "Talk to you later"
            r'catch.+you.+later',       # "Catch you later"
        ]
        
        # Block ALL casual responses
        casual_response_patterns = [
            r'i.+m.+(fine|good|okay|great|alright)',  # "I'm fine/good/okay/great"
            r'nothing.+much',           # "Nothing much"
            r'same.+here',              # "Same here"
            r'just.+(chatting|talking|chilling)',  # "Just chatting"
            r'not.+much',               # "Not much"
            r'pretty.+good',            # "Pretty good"
            r'doing.+(well|fine|good)', # "Doing well/fine/good"
            r'that.+s.+(cool|nice|great)', # "That's cool/nice/great"
            r'sounds.+(good|great|nice)', # "Sounds good/great/nice"
            r'i.+see',                  # "I see"
            r'oh.+(okay|ok|cool)',      # "Oh okay/ok/cool"
            r'gotcha',                  # "Gotcha"
            r'makes.+sense',            # "Makes sense"
        ]
        
        # Check ALL patterns
        all_patterns = question_to_buddy_patterns + greeting_patterns + casual_response_patterns
        
        for pattern in all_patterns:
            if re.search(pattern, text_lower):
                print(f"[SmartMemory] 🛡️ BLOCKED casual pattern '{pattern}' in: '{text}'")
                return True
        
        # Block if too short (less than 6 words for events)
        if len(text.split()) < 6:
            print(f"[SmartMemory] 🛡️ BLOCKED too short ({len(text.split())} words): '{text}'")
            return True
        
        # Block if it's a question (contains ?)
        if '?' in text:
            print(f"[SmartMemory] 🛡️ BLOCKED question mark detected: '{text}'")
            return True
        
        return False
    
    def _likely_contains_events(self, text: str) -> bool:
        """🎯 STRICT check - only allow if it DEFINITELY contains events"""
        text_lower = text.lower()
        
        # MUST contain time indicators - EXPANDED for casual time references AND FUTURE EVENTS
        time_indicators = [
            # Past time references
            'tomorrow', 'today', 'tonight', 'this week', 'next week', 'this weekend',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'this morning', 'this afternoon', 'this evening', 'later today',
            'in an hour', 'in a few hours', 'at ', 'o\'clock',
            'earlier today', 'earlier', 'just now', 'a while ago', 'this morning',
            'this afternoon', 'this evening', 'last night', 'yesterday', 'last week',
            'recently', 'just', 'before', 'after lunch', 'after work',
            # ENHANCED: Future time references for edge cases
            'next week wednesday', 'next wednesday', 'next friday', 'next monday',
            'tomorrow afternoon', 'tomorrow morning', 'tomorrow evening',
            'later this week', 'end of week', 'weekend', 'next month',
            'in a few days', 'in a week', 'soon', 'upcoming', 'planned',
            # ENHANCED: Time expressions
            '3pm', '3 pm', 'at 3', 'morning', 'afternoon', 'evening'
        ]
        
        has_time = any(indicator in text_lower for indicator in time_indicators)
        
        if not has_time:
            print(f"[SmartMemory] 🛡️ BLOCKED no time indicators: '{text}'")
            return False
        
        # MUST contain event keywords - ENHANCED for edge cases and future events
        event_keywords = [
            # Appointments (specific)
            'appointment', 'meeting', 'dentist', 'doctor', 'surgery', 'hospital',
            'interview', 'class', 'lesson', 'session', 'therapy', 'vet',
            'haircut', 'massage', 'nail', 'beauty', 'spa',
            
            # Social events (specific) - ENHANCED for edge cases
            'birthday', 'party', 'wedding', 'funeral', 'visit', 'visiting',
            'going to', 'seeing', 'dinner', 'lunch', 'coffee', 'movie',
            'concert', 'date', 'sleepover', 'hang out', 'play date',
            'niece', 'nephew', 'cousin', 'friend', 'family',
            
            # Work/school (specific)
            'work', 'job', 'school', 'exam', 'test', 'presentation', 
            'conference', 'training', 'orientation', 'review',
            
            # Travel (specific)
            'vacation', 'trip', 'flying', 'traveling', 'flight', 'train',
            'bus', 'driving to', 'pick up', 'drop off',
            
            # Places and activities (ENHANCED for edge cases)
            'went to', 'been to', 'was at', 'visited', 'stopped by',
            'mcdonalds', 'mcdonald', 'restaurant', 'cafe', 'shop', 'store',
            'mall', 'park', 'beach', 'gym', 'library', 'bank', 'church',
            'supermarket', 'grocery', 'pharmacy', 'gas station', 'cinema',
            'theater', 'museum', 'zoo', 'hospital', 'clinic', 'office',
            'coffee shop', 'food court', 'shopping center',
            
            # Activities worth remembering - ENHANCED
            'ate at', 'shopped at', 'bought from', 'had lunch at', 'had dinner at',
            'grabbed coffee', 'picked up', 'dropped off', 'met at', 'met with',
            'grabbed some food', 'grabbed food', 'got food', 'food at',
            'discuss', 'discussed', 'talking about', 'nervous about', 'excited about',
            
            # Actions with commitment - ENHANCED for edge cases
            'have to', 'need to', 'going for', 'scheduled', 'planned',
            'supposed to', 'meeting with', 'seeing', 'picking up',
            'off to', 'heading to', 'going to see'
        ]
        
        has_event = any(keyword in text_lower for keyword in event_keywords)
        
        if not has_event:
            print(f"[SmartMemory] 🛡️ BLOCKED no event keywords: '{text}'")
            return False
        
        # Additional validation - check for action verbs - ENHANCED for edge cases
        action_verbs = [
            'have', 'going', 'seeing', 'visiting', 'meeting', 'picking', 
            'dropping', 'starting', 'finishing', 'attending', 'scheduled',
            # ENHANCED: Add movement and activity verbs for edge cases
            'went', 'go', 'been', 'was', 'visited', 'stopped', 'ate', 'had',
            'bought', 'shopped', 'got', 'picked', 'dropped', 'met', 'saw',
            'grabbed', 'discussed', 'talked', 'planning', 'nervous', 'excited',
            # ENHANCED: Future planning verbs
            'off', 'heading', 'booked', 'scheduled', 'planned', 'arranging'
        ]
        
        has_action = any(verb in text_lower for verb in action_verbs)
        
        if not has_action:
            print(f"[SmartMemory] 🛡️ BLOCKED no action verbs: '{text}'")
            return False
        
        print(f"[SmartMemory] ✅ PASSED all event filters: '{text}'")
        return True
    
    def _contains_emotional_state(self, text: str) -> bool:
        """🎯 Check for emotional states worth remembering"""
        text_lower = text.lower()
        
        emotion_keywords = [
            'stressed', 'worried', 'anxious', 'nervous', 'scared', 'upset',
            'excited', 'thrilled', 'happy', 'glad', 'pleased', 'proud',
            'sad', 'depressed', 'down', 'frustrated', 'angry', 'annoyed',
            'tired', 'exhausted', 'overwhelmed', 'confused', 'lost',
            'hopeful', 'optimistic', 'confident', 'motivated', 'inspired'
        ]
        
        return any(emotion in text_lower for emotion in emotion_keywords)
    
    def _smart_detect_events(self, text: str) -> List[Dict]:
        """🧠 Use Hermes 3 Pro Mistral to intelligently detect events - BULLETPROOF FILTERED"""
        
        # ✅ TRIPLE FILTERING SYSTEM - BULLETPROOF!
        
        # Filter 1: Block casual conversation
        if self._is_casual_conversation(text):
            return []
        
        # Filter 2: Must contain events OR emotions
        has_events = self._likely_contains_events(text)
        has_emotions = self._contains_emotional_state(text)
        
        if not has_events and not has_emotions:
            print(f"[SmartMemory] 🛡️ BLOCKED no events or emotions: '{text}'")
            return []
        
        # Filter 3: Final validation - must be substantial
        if len(text.split()) < 5:
            print(f"[SmartMemory] 🛡️ BLOCKED too short for events: '{text}'")
            return []
        
        # If we get here, it's worth LLM processing
        print(f"[SmartMemory] 🎯 APPROVED for LLM processing: '{text}'")
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H:%M')
        
        # Smart prompt for event detection
        detection_prompt = f"""You are a smart memory assistant. Analyze this user message and extract any events, appointments, or life situations that should be remembered.

Current date: {current_date}
Current time: {current_time}
User message: "{text}"

Extract events in this exact JSON format (return empty array if no events):
[
  {{
    "type": "appointment|life_event|highlight",
    "topic": "brief_description",
    "date": "YYYY-MM-DD",
    "time": "HH:MM" or null,
    "emotion": "happy|excited|stressful|sensitive|casual|supportive",
    "priority": "high|medium|low",
    "original_text": "{text}"
  }}
]

Guidelines:
- "appointment": Time-specific events (dentist, meeting, class)
- "life_event": Emotional/social events (birthdays, visits, funerals)  
- "highlight": General feelings/thoughts to remember
- Calculate dates: tomorrow = {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}
- Be smart about natural language: "going to Emily's tomorrow, it's her birthday" = birthday visit
- Emotion should match the event type
- Priority: high=urgent/sensitive, medium=social/fun, low=routine
- ONLY extract if it's a REAL event or emotional state worth remembering
- DO NOT extract casual conversation, greetings, or questions

Examples:
"I have dentist tomorrow at 2PM" → appointment, dentist, tomorrow, 14:00, stressful, medium
"Going to Emily's tomorrow, it's her birthday" → life_event, Emily's birthday visit, tomorrow, happy, medium
"I'm really stressed about work" → highlight, work stress, today, supportive, low

Return only valid JSON array:"""

        try:
            # ✅ TOKEN OPTIMIZATION: Use optimized event detection
            from ai.llm_optimized import detect_events_optimized
            
            events = detect_events_optimized(text, current_date)
            
            # Validate and enhance events
            validated_events = []
            for event in events:
                if self._validate_event(event):
                    enhanced_event = self._enhance_event(event)
                    validated_events.append(enhanced_event)
            
            print(f"[SmartMemory] 🧠 Optimized detected {len(validated_events)} events from: '{text}'")
            
            # ✅ CRITICAL FIX: If optimized detection returns no events, try fallback
            if len(validated_events) == 0:
                print(f"[SmartMemory] 🔄 No events from optimized detection, trying fallback...")
                fallback_events = self._fallback_detection(current_date, text)
                if fallback_events:
                    print(f"[SmartMemory] ✅ Fallback detected {len(fallback_events)} events")
                    return fallback_events
            
            return validated_events
            
        except ImportError:
            print(f"[SmartMemory] ⚠️ Optimized detection not available, using fallback")
            return self._detect_events_fallback(text, current_date)
        except Exception as e:
            print(f"[SmartMemory] ❌ Optimized detection error: {e}")
            return self._detect_events_fallback(text, current_date)
    
    def _detect_events_fallback(self, text: str, current_date: str) -> List[Dict]:
        """Fallback to original LLM event detection"""
        
        current_time = datetime.now().strftime('%H:%M')
        
        # Smart prompt for event detection (original)
        detection_prompt = f"""You are a smart memory assistant. Analyze this user message and extract any events, appointments, or life situations that should be remembered.

Current date: {current_date}
Current time: {current_time}
User message: "{text}"

Extract events in this exact JSON format (return empty array if no events):
[
  {{
    "type": "appointment|life_event|highlight",
    "topic": "brief_description",
    "date": "YYYY-MM-DD",
    "time": "HH:MM" or null,
    "emotion": "happy|excited|stressful|sensitive|casual|supportive",
    "priority": "high|medium|low",
    "original_text": "{text}"
  }}
.]

Guidelines:
- "appointment": Time-specific events (dentist, meeting, class)
- "life_event": Emotional/social events (birthdays, visits, funerals)  
- "highlight": General feelings/thoughts to remember
- Calculate dates: tomorrow = {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}
- Be smart about natural language: "going to Emily's tomorrow, it's her birthday" = birthday visit
- Emotion should match the event type
- Priority: high=urgent/sensitive, medium=social/fun, low=routine
- ONLY extract if it's a REAL event or emotional state worth remembering
- DO NOT extract casual conversation, greetings, or questions

Examples:
"I have dentist tomorrow at 2PM" → appointment, dentist, tomorrow, 14:00, stressful, medium
"Going to Emily's tomorrow, it's her birthday" → life_event, Emily's birthday visit, tomorrow, happy, medium
"I'm really stressed about work" → highlight, work stress, today, supportive, low

Return only valid JSON array:"""

        try:
            # Get LLM response
            messages = [
                {"role": "system", "content": "You are a precise JSON extraction assistant. Return only valid JSON arrays. Extract ONLY real events, appointments, or emotional states worth remembering."},
                {"role": "user", "content": detection_prompt}
            ]
            
            llm_response = ask_kobold(messages, max_tokens=300)
            
            # Clean and parse JSON response
            json_text = self._extract_json_from_response(llm_response)
            
            if json_text:
                events = json.loads(json_text)
                
                # Validate and enhance events
                validated_events = []
                for event in events:
                    if self._validate_event(event):
                        enhanced_event = self._enhance_event(event)
                        validated_events.append(enhanced_event)
                
                print(f"[SmartMemory] 🧠 Fallback detected {len(validated_events)} events from: '{text}'")
                return validated_events
            
        except Exception as e:
            print(f"[SmartMemory] ❌ Fallback analysis error: {e}")
        
        # Enhanced fallback to regex patterns for edge cases
        return self._fallback_detection(current_date, text)
    
    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """Extract JSON array from LLM response"""
        try:
            # Look for JSON array in response
            import re
            
            # Find JSON array pattern
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                return json_match.group(0)
            
            # Look for individual JSON objects and wrap in array
            obj_matches = re.findall(r'\{.*?\}', response, re.DOTALL)
            if obj_matches:
                return '[' + ','.join(obj_matches) + ']'
            
        except Exception as e:
            print(f"[SmartMemory] JSON extraction error: {e}")
        
        return None
    
    def _validate_event(self, event: Dict) -> bool:
        """Validate event has required fields"""
        required_fields = ['type', 'topic', 'date', 'emotion']
        return all(field in event for field in required_fields)
    
    def _enhance_event(self, event: Dict) -> Dict:
        """Enhance event with additional fields"""
        current_time = datetime.now().isoformat()
        
        enhanced = {
            'topic': event['topic'],
            'date': event['date'],
            'time': event.get('time'),
            'emotion': event['emotion'],
            'status': 'pending',
            'type': event['type'],
            'priority': event.get('priority', 'medium'),
            'created': current_time,
            'original_text': event.get('original_text', ''),
            'detected_by': 'llm'
        }
        
        return enhanced
    
    def _fallback_detection(self, current_date: str, text: str) -> List[Dict]:
        """Enhanced fallback detection for critical patterns and edge cases"""
        events = []
        text_lower = text.lower()
        tomorrow_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        next_week_dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7, 14)]
        
        # ENHANCED: Appointment patterns (with time and emotional context)
        appointment_pattern = r'(?:dentist|doctor|appointment|meeting).+(?:tomorrow|today|next week|wednesday|monday|tuesday|thursday|friday).+(?:\d{1,2}(?::\d{2})?(?:\s?(?:am|pm|AM|PM))?|\d{1,2}\s?(?:am|pm|AM|PM))'
        if re.search(appointment_pattern, text_lower):
            # Determine date
            if 'tomorrow' in text_lower:
                event_date = tomorrow_date
            elif 'next week' in text_lower or any(day in text_lower for day in ['wednesday', 'monday', 'tuesday', 'thursday', 'friday']):
                event_date = next_week_dates[2] if 'wednesday' in text_lower else next_week_dates[0]  # Default to next week start
            else:
                event_date = current_date
                
            # Detect emotional context
            emotion = 'casual'
            if any(word in text_lower for word in ['nervous', 'worried', 'scared', 'anxious']):
                emotion = 'stressful'
            elif any(word in text_lower for word in ['excited', 'looking forward', 'happy']):
                emotion = 'happy'
                
            events.append({
                'type': 'appointment',
                'topic': 'dentist appointment' if 'dentist' in text_lower else 'appointment',
                'date': event_date,
                'emotion': emotion,
                'status': 'pending',
                'created': datetime.now().isoformat(),
                'original_text': text,
                'detected_by': 'enhanced_fallback'
            })
        
        # ENHANCED: Birthday with specific person and date
        birthday_pattern = r"(?:(?:i'm\s+)?(?:off\s+to|going\s+to)?(?:\s+my\s+)?(\w+)(?:'s)?\s+birthday).+(?:next\s+week\s+)?(?:wednesday|monday|tuesday|thursday|friday|saturday|sunday|tomorrow|today)"
        birthday_match = re.search(birthday_pattern, text_lower)
        if birthday_match:
            person = birthday_match.group(1)
            # Determine date
            if 'next week wednesday' in text_lower or 'wednesday' in text_lower:
                event_date = next_week_dates[2]  # Wednesday of next week
            elif 'tomorrow' in text_lower:
                event_date = tomorrow_date
            else:
                event_date = current_date
                
            events.append({
                'type': 'life_event',
                'topic': f'{person}_birthday_visit',
                'date': event_date,
                'emotion': 'happy',
                'status': 'pending',
                'created': datetime.now().isoformat(),
                'original_text': text,
                'detected_by': 'enhanced_fallback'
            })
        
        # ENHANCED: Food/restaurant visits with companions
        food_pattern = r'(?:grabbed|got|had|ate)\s+(?:some\s+)?food\s+at\s+(\w+)(?:\s+with\s+(\w+))?(?:\s+earlier)?'
        food_match = re.search(food_pattern, text_lower)
        if food_match:
            place = food_match.group(1)
            companion = food_match.group(2) if food_match.group(2) else None
            
            topic = f'visited_{place.lower()}'
            if companion:
                topic += f'_with_{companion.lower()}'
                
            events.append({
                'type': 'life_event',
                'topic': topic,
                'date': current_date if 'earlier' in text_lower or 'today' in text_lower else current_date,
                'emotion': 'casual',
                'status': 'pending',
                'created': datetime.now().isoformat(),
                'original_text': text,
                'detected_by': 'enhanced_fallback'
            })
        
        # ENHANCED: Alternative food patterns for different phrasings
        food_alt_pattern = r'(?:grabbed|got|had|ate)\s+(?:some\s+)?(?:food|lunch|dinner|breakfast)\s+(?:at\s+)?(?:the\s+)?(\w+\'?\w*)\s+with\s+(\w+)'
        food_alt_match = re.search(food_alt_pattern, text_lower)
        if food_alt_match and not food_match:  # Only if first pattern didn't match
            place = food_alt_match.group(1).replace("'", "")  # Remove apostrophes
            companion = food_alt_match.group(2)
            
            topic = f'visited_{place.lower()}_with_{companion.lower()}'
                
            events.append({
                'type': 'life_event',
                'topic': topic,
                'date': current_date,
                'emotion': 'casual',
                'status': 'pending',
                'created': datetime.now().isoformat(),
                'original_text': text,
                'detected_by': 'enhanced_fallback_alt'
            })
        
        # ENHANCED: Meeting patterns with specific people
        meeting_pattern = r'(?:met|meeting)\s+(\w+)\s+at\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:this\s+)?(?:morning|afternoon|evening|today|yesterday)'
        meeting_match = re.search(meeting_pattern, text_lower)
        if meeting_match:
            person = meeting_match.group(1)
            place = meeting_match.group(2)
            
            events.append({
                'type': 'life_event',
                'topic': f'met_{person.lower()}_at_{place.lower().replace(" ", "_")}',
                'date': current_date,
                'emotion': 'casual',
                'status': 'pending',
                'created': datetime.now().isoformat(),
                'original_text': text,
                'detected_by': 'enhanced_fallback'
            })
        
        # ENHANCED: Shop visits with companions
        shop_pattern = r'(?:we|i)\s+went\s+to\s+(?:the\s+)?shop\s+(?:last\s+night|yesterday|earlier|today)'
        if re.search(shop_pattern, text_lower):
            events.append({
                'type': 'life_event',
                'topic': 'visited_shop',
                'date': current_date if any(time_ref in text_lower for time_ref in ['today', 'earlier']) else (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'emotion': 'casual',
                'status': 'pending',
                'created': datetime.now().isoformat(),
                'original_text': text,
                'detected_by': 'enhanced_fallback'
            })
        
        return events
    
    def load_memory(self, filename: str) -> List[Dict]:
        """Load memory file"""
        file_path = os.path.join(self.memory_dir, filename)
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"[SmartMemory] ⚠️ Error loading {filename}: {e}")
            return []
    
    def save_memory(self, data: List[Dict], filename: str):
        """Save memory file"""
        file_path = os.path.join(self.memory_dir, filename)
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[SmartMemory] ❌ Error saving {filename}: {e}")
    
    def check_for_natural_context_response(self) -> Optional[str]:
        """🎯 Check if we should naturally bring up memories"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Check appointments first (time-sensitive)
        response = self._check_smart_appointments(today)
        if response:
            return response
        
        # Check life events 
        response = self._check_smart_life_events(today)
        if response:
            return response
        
        # Check conversation highlights (occasionally)
        if random.random() < 0.2:  # 20% chance
            response = self._check_smart_highlights()
            if response:
                return response
        
        return None
    
    def _check_smart_appointments(self, today: str) -> Optional[str]:
        """📅 Check smart appointments"""
        for appointment in self.appointments:
            if appointment['date'] == today and appointment['status'] == 'pending':
                topic = appointment['topic']
                time_str = appointment.get('time', '')
                
                context_key = f"appointment_{appointment['date']}_{topic}"
                if context_key in self.context_used_this_session:
                    continue
                
                self.context_used_this_session.add(context_key)
                
                if time_str:
                    if 'dentist' in topic.lower():
                        responses = [
                            f"Yo! Just a heads up — {topic} at {time_str}. You ready for the drill?",
                            f"Hey, don't forget your {topic} at {time_str}! Try not to chicken out"
                        ]
                    else:
                        responses = [
                            f"Heads up — {topic} at {time_str} today. You got this!",
                            f"Don't forget your {topic} at {time_str}!"
                        ]
                else:
                    responses = [
                        f"Hey, you've got {topic} today. How you feeling about it?",
                        f"Don't forget about {topic} today!"
                    ]
                
                appointment['status'] = 'reminded'
                self.save_memory(self.appointments, 'smart_appointments.json')
                
                return random.choice(responses)
        
        return None
    
    def _check_smart_life_events(self, today: str) -> Optional[str]:
        """📝 Check smart life events"""
        for event in self.life_events:
            if event['date'] == today and event['status'] == 'pending':
                topic = event['topic']
                emotion = event['emotion']
                
                context_key = f"life_event_{event['date']}_{topic}"
                if context_key in self.context_used_this_session:
                    continue
                
                self.context_used_this_session.add(context_key)
                
                if emotion == 'sensitive':
                    responses = [
                        f"Hey, how'd the {topic} go? You alright?",
                        f"Just checking — how was the {topic}? You doing okay?"
                    ]
                elif emotion in ['happy', 'excited']:
                    responses = [
                        f"Yooo! How was the {topic}? Tell me everything!",
                        f"How'd the {topic} go?! Was it awesome?"
                    ]
                elif emotion == 'stressful':
                    responses = [
                        f"So how'd the {topic} go? You survive?",
                        f"How was the {topic}? Hope it went better than expected!"
                    ]
                else:
                    responses = [
                        f"How'd the {topic} go?",
                        f"So, how was your {topic}?"
                    ]
                
                event['status'] = 'followed_up'
                self.save_memory(self.life_events, 'smart_life_events.json')
                
                return random.choice(responses)
        
        return None
    
    def _check_smart_highlights(self) -> Optional[str]:
        """💬 Check smart conversation highlights"""
        recent_highlights = []
        cutoff_date = datetime.now() - timedelta(days=3)
        
        for highlight in self.conversation_highlights:
            created = datetime.fromisoformat(highlight['created'])
            if created >= cutoff_date and highlight['status'] == 'pending':
                context_key = f"highlight_{highlight['topic']}_{highlight['created'][:10]}"
                if context_key not in self.context_used_this_session:
                    recent_highlights.append(highlight)
        
        if recent_highlights:
            highlight = random.choice(recent_highlights)
            topic = highlight['topic']
            
            context_key = f"highlight_{topic}_{highlight['created'][:10]}"
            self.context_used_this_session.add(context_key)
            
            responses = [
                f"Hey, how's that {topic} situation going?",
                f"Any updates on {topic}?"
            ]
            
            highlight['status'] = 'followed_up'
            self.save_memory(self.conversation_highlights, 'smart_highlights.json')
            
            return random.choice(responses)
        
        return None
    
    def reset_session_context(self):
        """Reset session context"""
        self.context_used_this_session.clear()
        print(f"[SmartMemory] 🔄 Session context reset for {self.username}")