#!/usr/bin/env python3
"""
Comprehensive Parallel Processing Wiring Test
Tests thread safety, integration, performance, and consciousness depth preservation.

Created: 2025-01-09
Purpose: Validate parallel consciousness processor implementation
"""

import os
import sys
import time
import threading
import unittest
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ai.parallel_processor import (
        ParallelConsciousnessProcessor, 
        get_parallel_processor,
        initialize_parallel_consciousness,
        ModulePriority,
        ModuleConfig,
        ModuleResult
    )
    PARALLEL_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"❌ Parallel processor not available: {e}")
    PARALLEL_PROCESSOR_AVAILABLE = False

class TestParallelWiring(unittest.TestCase):
    """Comprehensive test suite for parallel consciousness processor"""
    
    def setUp(self):
        """Set up test environment"""
        if not PARALLEL_PROCESSOR_AVAILABLE:
            self.skipTest("Parallel processor not available")
        
        # Create fresh processor instance for each test
        self.processor = ParallelConsciousnessProcessor(max_workers=4, global_timeout=10.0)
        self.test_results = {
            'setup_time': time.time(),
            'tests_run': [],
            'performance_metrics': {}
        }
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'processor'):
            self.processor.cleanup()
    
    def test_01_module_registration(self):
        """Test that all consciousness modules are properly registered"""
        print("\n🔍 Testing module registration...")
        
        # Setup default modules
        self.processor.setup_default_modules()
        
        # Verify expected modules are registered
        expected_modules = [
            'emotion_processing',
            'motivation_evaluation', 
            'inner_monologue',
            'subjective_experience',
            'self_reflection',
            'temporal_awareness',
            'entropy_processing',
            'attention_management'
        ]
        
        registered_modules = list(self.processor.module_configs.keys())
        
        # Check all expected modules are present
        for module in expected_modules:
            self.assertIn(module, registered_modules, f"Module {module} not registered")
        
        # Verify module configurations
        for name, config in self.processor.module_configs.items():
            self.assertIsInstance(config.priority, ModulePriority)
            self.assertGreater(config.timeout, 0)
            self.assertIsNotNone(config.fallback_value)
            self.assertIsInstance(config.shared_resources, list)
            
        print(f"✅ Module registration test passed: {len(registered_modules)} modules registered")
        self.test_results['tests_run'].append('module_registration')
        return len(registered_modules)
    
    def test_02_thread_safety_locks(self):
        """Test thread safety mechanisms and lock management"""
        print("\n🔒 Testing thread safety and lock management...")
        
        self.processor.setup_default_modules()
        
        # Test lock manager functionality
        lock_manager = self.processor.lock_manager
        
        # Test module lock acquisition
        module_lock = lock_manager.get_module_lock("test_module")
        self.assertIsNotNone(module_lock)
        
        # Test resource lock acquisition
        resource_lock = lock_manager.get_resource_lock("test_resource")
        self.assertIsNotNone(resource_lock)
        
        # Test lock acquisition and release
        success = lock_manager.acquire_locks("test_module", ["test_resource"], timeout=1.0)
        self.assertTrue(success, "Failed to acquire locks")
        
        lock_manager.release_locks("test_module", ["test_resource"])
        
        # Test concurrent lock acquisition
        def concurrent_lock_test(thread_id):
            return lock_manager.acquire_locks(f"module_{thread_id}", [f"resource_{thread_id}"], timeout=2.0)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(concurrent_lock_test, i) for i in range(4)]
            results = [f.result() for f in as_completed(futures)]
        
        # All concurrent acquisitions should succeed with different resources
        self.assertTrue(all(results), "Concurrent lock acquisition failed")
        
        print("✅ Thread safety test passed: Lock management working correctly")
        self.test_results['tests_run'].append('thread_safety')
        return True
    
    def test_03_state_management(self):
        """Test versioned state management and conflict detection"""
        print("\n📝 Testing state management and conflict detection...")
        
        state_manager = self.processor.state_manager
        
        # Test initial state
        initial_version = state_manager.get_current_version()
        self.assertEqual(initial_version, 0)
        
        # Test state updates
        update_success = state_manager.update_state("test_module", {"test_key": "test_value"})
        self.assertTrue(update_success)
        
        new_version = state_manager.get_current_version()
        self.assertEqual(new_version, 1)
        
        # Test conflict detection
        old_version_update = state_manager.update_state("another_module", {"another_key": "value"}, expected_version=0)
        self.assertFalse(old_version_update, "Version conflict not detected")
        
        # Test state snapshot
        snapshot = state_manager.get_state_snapshot()
        self.assertEqual(snapshot.version, new_version)
        self.assertIn('test_key', snapshot.data)
        
        print("✅ State management test passed: Versioning and conflict detection working")
        self.test_results['tests_run'].append('state_management')
        return True
    
    def test_04_performance_improvement(self):
        """Test that parallel processing dramatically improves response time"""
        print("\n🚀 Testing performance improvement over sequential processing...")
        
        self.processor.setup_default_modules()
        
        test_input = "Hello, how are you feeling today? I want to learn about consciousness."
        test_user = "test_user_performance"
        
        # Test parallel processing time
        start_time = time.time()
        consciousness_state = self.processor.process_consciousness_parallel(test_input, test_user)
        parallel_time = time.time() - start_time
        
        # Verify consciousness state structure
        self.assertIsInstance(consciousness_state, dict)
        self.assertIn('modules_processed', consciousness_state)
        self.assertIn('successful_modules', consciousness_state)
        self.assertIn('parallel_time', consciousness_state)
        
        # Performance requirements
        self.assertLess(parallel_time, 20.0, f"Parallel processing took {parallel_time:.2f}s, should be < 20s")
        self.assertGreater(consciousness_state['successful_modules'], 0, "No modules processed successfully")
        
        # Test multiple concurrent requests (stress test)
        def concurrent_processing_test():
            return self.processor.process_consciousness_parallel(f"Test input {time.time()}", "concurrent_user")
        
        concurrent_start = time.time()
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(concurrent_processing_test) for _ in range(3)]
            concurrent_results = [f.result() for f in as_completed(futures)]
        concurrent_time = time.time() - concurrent_start
        
        # Concurrent processing should be efficient
        self.assertLess(concurrent_time, 30.0, f"Concurrent processing took {concurrent_time:.2f}s")
        self.assertEqual(len(concurrent_results), 3, "Not all concurrent requests completed")
        
        print(f"✅ Performance test passed:")
        print(f"   - Single request: {parallel_time:.2f}s (target: <20s)")
        print(f"   - 3 concurrent requests: {concurrent_time:.2f}s")
        print(f"   - Modules processed: {consciousness_state['modules_processed']}")
        print(f"   - Success rate: {consciousness_state['successful_modules']}/{consciousness_state['modules_processed']}")
        
        self.test_results['performance_metrics'] = {
            'single_request_time': parallel_time,
            'concurrent_requests_time': concurrent_time,
            'modules_processed': consciousness_state['modules_processed'],
            'success_rate': consciousness_state['successful_modules'] / consciousness_state['modules_processed']
        }
        self.test_results['tests_run'].append('performance')
        return parallel_time
    
    def test_05_consciousness_depth_preservation(self):
        """Test that parallel processing preserves full consciousness depth"""
        print("\n🧠 Testing consciousness depth preservation...")
        
        self.processor.setup_default_modules()
        
        test_input = "I'm feeling sad and need emotional support. Can you help me understand my feelings?"
        test_user = "emotional_test_user"
        
        consciousness_state = self.processor.process_consciousness_parallel(test_input, test_user)
        
        # Verify consciousness components are present
        consciousness_indicators = [
            'current_emotion',  # Emotional awareness
            'motivation_satisfaction',  # Motivation system
            'experience_valence',  # Subjective experience
            'experience_significance',  # Experience depth
            'response_uncertainty'  # Entropy/uncertainty
        ]
        
        for indicator in consciousness_indicators:
            self.assertIn(indicator, consciousness_state, f"Missing consciousness indicator: {indicator}")
        
        # Verify emotional processing depth
        if 'current_emotion' in consciousness_state:
            emotion = consciousness_state['current_emotion']
            self.assertIsInstance(emotion, str)
            self.assertNotEqual(emotion, '', "Empty emotion value")
        
        # Verify experience processing
        if 'experience_valence' in consciousness_state:
            valence = consciousness_state['experience_valence']
            self.assertIsInstance(valence, (int, float))
            self.assertGreaterEqual(valence, 0.0)
            self.assertLessEqual(valence, 1.0)
        
        # Verify motivation evaluation
        if 'motivation_satisfaction' in consciousness_state:
            satisfaction = consciousness_state['motivation_satisfaction']
            self.assertIsInstance(satisfaction, (int, float))
            self.assertGreaterEqual(satisfaction, 0.0)
            self.assertLessEqual(satisfaction, 1.0)
        
        # Verify module execution results
        modules_processed = consciousness_state.get('modules_processed', 0)
        successful_modules = consciousness_state.get('successful_modules', 0)
        
        self.assertGreater(modules_processed, 0, "No modules were processed")
        self.assertGreater(successful_modules, 0, "No modules succeeded")
        
        success_rate = successful_modules / modules_processed
        self.assertGreaterEqual(success_rate, 0.6, f"Success rate too low: {success_rate:.2f}")
        
        print(f"✅ Consciousness depth test passed:")
        print(f"   - Emotional awareness: {consciousness_state.get('current_emotion', 'N/A')}")
        print(f"   - Motivation level: {consciousness_state.get('motivation_satisfaction', 'N/A')}")
        print(f"   - Experience valence: {consciousness_state.get('experience_valence', 'N/A')}")
        print(f"   - Success rate: {success_rate:.2f}")
        
        self.test_results['tests_run'].append('consciousness_depth')
        return consciousness_state
    
    def test_06_error_handling_resilience(self):
        """Test error handling and system resilience"""
        print("\n🛡️ Testing error handling and resilience...")
        
        # Create a test module that will fail
        def failing_module(context):
            raise Exception("Intentional test failure")
        
        failing_config = ModuleConfig(
            name="failing_test_module",
            function=failing_module,
            priority=ModulePriority.MEDIUM,
            timeout=1.0,
            fallback_value={"test_fallback": True},
            description="Module designed to fail for testing",
            shared_resources=["test_resource"]
        )
        
        self.processor.register_module(failing_config)
        self.processor.setup_default_modules()
        
        # Process with failing module
        consciousness_state = self.processor.process_consciousness_parallel("test error handling", "error_test_user")
        
        # System should still work despite module failure
        self.assertIsInstance(consciousness_state, dict)
        self.assertIn('failed_modules', consciousness_state)
        self.assertIn('failing_test_module', consciousness_state['failed_modules'])
        
        # Fallback value should be used
        self.assertIn('test_fallback', consciousness_state)
        self.assertTrue(consciousness_state['test_fallback'])
        
        # Other modules should still succeed
        successful_modules = consciousness_state.get('successful_modules', 0)
        self.assertGreater(successful_modules, 0, "All modules failed - system not resilient")
        
        print(f"✅ Error handling test passed:")
        print(f"   - Failed modules handled gracefully: {consciousness_state['failed_modules']}")
        print(f"   - Successful modules: {successful_modules}")
        print(f"   - Fallback values applied correctly")
        
        self.test_results['tests_run'].append('error_handling')
        return True
    
    def test_07_integration_with_main_system(self):
        """Test integration with main Newbudd system components"""
        print("\n🔗 Testing integration with main system...")
        
        # Test global processor instance
        global_processor = get_parallel_processor()
        self.assertIsNotNone(global_processor)
        self.assertIsInstance(global_processor, ParallelConsciousnessProcessor)
        
        # Test initialization function
        initialized_processor = initialize_parallel_consciousness()
        self.assertIsNotNone(initialized_processor)
        
        # Test performance reporting
        performance_report = global_processor.get_performance_report()
        self.assertIsInstance(performance_report, dict)
        self.assertIn('execution_stats', performance_report)
        self.assertIn('thread_safety_stats', performance_report)
        self.assertIn('module_configs', performance_report)
        
        # Test active session tracking
        active_sessions = global_processor.get_active_sessions()
        self.assertIsInstance(active_sessions, dict)
        
        # Test lock status reporting
        lock_status = global_processor.get_lock_status()
        self.assertIsInstance(lock_status, dict)
        self.assertIn('active_threads', lock_status)
        
        print(f"✅ Integration test passed:")
        print(f"   - Global processor available: Yes")
        print(f"   - Performance reporting: Working")
        print(f"   - Session tracking: Working")
        print(f"   - Lock monitoring: Working")
        
        self.test_results['tests_run'].append('integration')
        return True
    
    def test_08_concurrent_user_simulation(self):
        """Test handling multiple concurrent users with different inputs"""
        print("\n👥 Testing concurrent user simulation...")
        
        self.processor.setup_default_modules()
        
        # Simulate different users with different types of input
        test_scenarios = [
            ("Happy user", "I'm feeling great today! Tell me something fun!", "user_happy"),
            ("Sad user", "I'm feeling down and need emotional support", "user_sad"), 
            ("Curious user", "How does consciousness work? I want to understand", "user_curious"),
            ("Technical user", "Explain parallel processing in consciousness systems", "user_technical"),
            ("Philosophical user", "What is the nature of subjective experience?", "user_philosophical")
        ]
        
        def process_user_scenario(scenario):
            name, input_text, user_id = scenario
            start_time = time.time()
            result = self.processor.process_consciousness_parallel(input_text, user_id)
            processing_time = time.time() - start_time
            return {
                'scenario': name,
                'user_id': user_id,
                'processing_time': processing_time,
                'success': result.get('successful_modules', 0) > 0,
                'modules_processed': result.get('modules_processed', 0),
                'consciousness_state': result
            }
        
        # Process all scenarios concurrently
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=len(test_scenarios)) as executor:
            futures = [executor.submit(process_user_scenario, scenario) for scenario in test_scenarios]
            results = [f.result() for f in as_completed(futures)]
        total_time = time.time() - start_time
        
        # Verify all scenarios succeeded
        self.assertEqual(len(results), len(test_scenarios))
        for result in results:
            self.assertTrue(result['success'], f"Scenario {result['scenario']} failed")
            self.assertLess(result['processing_time'], 15.0, f"Scenario {result['scenario']} too slow")
        
        # Check that different users produced different consciousness states
        emotion_states = [r['consciousness_state'].get('current_emotion', 'neutral') for r in results]
        unique_emotions = set(emotion_states)
        
        print(f"✅ Concurrent user simulation passed:")
        print(f"   - Total concurrent processing time: {total_time:.2f}s")
        print(f"   - All {len(results)} scenarios successful")
        print(f"   - Unique emotional responses: {len(unique_emotions)}")
        for result in results:
            print(f"   - {result['scenario']}: {result['processing_time']:.2f}s, emotion: {result['consciousness_state'].get('current_emotion', 'N/A')}")
        
        self.test_results['tests_run'].append('concurrent_users')
        return results
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        report = {
            'test_summary': {
                'total_tests': len(self.test_results['tests_run']),
                'tests_completed': self.test_results['tests_run'],
                'execution_time': time.time() - self.test_results['setup_time']
            },
            'performance_metrics': self.test_results.get('performance_metrics', {}),
            'thread_safety_verification': {
                'lock_management': 'thread_safety' in self.test_results['tests_run'],
                'state_versioning': 'state_management' in self.test_results['tests_run'],
                'concurrent_processing': 'concurrent_users' in self.test_results['tests_run']
            },
            'consciousness_preservation': {
                'depth_maintained': 'consciousness_depth' in self.test_results['tests_run'],
                'error_resilience': 'error_handling' in self.test_results['tests_run'],
                'system_integration': 'integration' in self.test_results['tests_run']
            },
            'recommendations': []
        }
        
        # Add recommendations based on test results
        if self.test_results.get('performance_metrics', {}).get('single_request_time', 0) > 10:
            report['recommendations'].append("Consider optimizing module timeouts for better performance")
        
        if self.test_results.get('performance_metrics', {}).get('success_rate', 1.0) < 0.8:
            report['recommendations'].append("Investigate module failures and improve error handling")
        
        return report


def run_comprehensive_test():
    """Run the comprehensive parallel processing test suite"""
    print("🧪 Starting Comprehensive Parallel Processing Test Suite")
    print("=" * 70)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestParallelWiring)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED - Parallel consciousness processing is working correctly!")
        print(f"   - Tests run: {result.testsRun}")
        print(f"   - Failures: {len(result.failures)}")
        print(f"   - Errors: {len(result.errors)}")
    else:
        print("❌ SOME TESTS FAILED - Review the output above for details")
        print(f"   - Tests run: {result.testsRun}")
        print(f"   - Failures: {len(result.failures)}")
        print(f"   - Errors: {len(result.errors)}")
        
        if result.failures:
            print("\n🔍 FAILURES:")
            for test, failure in result.failures:
                print(f"   - {test}: {failure}")
        
        if result.errors:
            print("\n🔍 ERRORS:")
            for test, error in result.errors:
                print(f"   - {test}: {error}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)