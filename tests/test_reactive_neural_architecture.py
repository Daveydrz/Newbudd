#!/usr/bin/env python3
"""
Comprehensive Test Suite for Reactive Neural Architecture

Tests all components of the reactive neural architecture:
1. Event-Driven Nervous System
2. Async Neural Pathways
3. Parallel Processing Cortex (Work-Stealing)
4. Integration Layer
5. Optimized Memory Management
6. Hybrid Consciousness Processing

Created: 2025-01-09
Purpose: Validate reactive neural architecture implementation
"""

import os
import sys
import asyncio
import time
import threading
import unittest
import logging
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestReactiveNeuralArchitecture(unittest.TestCase):
    """Comprehensive test suite for reactive neural architecture"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"🧪 Starting test: {self._testMethodName}")
        print(f"{'='*60}")
    
    def tearDown(self):
        """Clean up after test"""
        test_duration = time.time() - self.test_start_time
        print(f"✅ Test completed in {test_duration:.3f}s")
        print(f"{'='*60}\n")
    
    def test_001_event_bus_basic_functionality(self):
        """Test 1: Event Bus Basic Functionality"""
        print("🚌 Testing Event Bus basic pub/sub functionality...")
        
        try:
            from ai.reactive_neural_architecture import EventBus, NeuralEvent, EventType, EventPriority
            
            # Create event bus
            event_bus = EventBus()
            
            # Test data
            received_events = []
            
            # Define subscriber
            def test_subscriber(event):
                received_events.append(event)
                print(f"📥 Received event: {event.type.name}")
            
            # Subscribe to events
            event_bus.subscribe("USER_INPUT", test_subscriber)
            
            # Create and publish test event
            test_event = NeuralEvent(
                type=EventType.USER_INPUT,
                priority=EventPriority.HIGH,
                source="test_source",
                data={'text': 'Hello, test!', 'user': 'test_user'}
            )
            
            # Publish synchronously
            event_bus.publish_sync(test_event)
            
            # Wait briefly for processing
            time.sleep(0.1)
            
            # Verify event was received
            self.assertEqual(len(received_events), 1)
            self.assertEqual(received_events[0].type, EventType.USER_INPUT)
            self.assertEqual(received_events[0].data['text'], 'Hello, test!')
            
            print(f"✅ Event Bus test passed: {len(received_events)} events processed")
            
        except Exception as e:
            self.fail(f"Event Bus test failed: {e}")
    
    def test_002_async_neural_pathways(self):
        """Test 2: Async Neural Pathways"""
        print("🧠 Testing Async Neural Pathways functionality...")
        
        try:
            from ai.reactive_neural_architecture import AsyncNeuralPathways
            
            async def run_async_test():
                # Create async pathways
                pathways = AsyncNeuralPathways(max_concurrent=5)
                
                # Test operation
                def test_operation(value, delay=0.1):
                    time.sleep(delay)
                    return value * 2
                
                # Execute operations
                operations = []
                for i in range(3):
                    op = pathways.execute_compute_operation(
                        f"test_op_{i}",
                        test_operation,
                        i + 1,
                        delay=0.05
                    )
                    operations.append(op)
                
                # Wait for results
                results = await asyncio.gather(*operations)
                
                # Verify results
                expected = [2, 4, 6]  # [1*2, 2*2, 3*2]
                self.assertEqual(results, expected)
                
                # Get statistics
                stats = await pathways.get_stats()
                self.assertGreaterEqual(stats['total_operations'], 3)
                self.assertGreaterEqual(stats['successful_operations'], 3)
                
                print(f"✅ Async Neural Pathways test passed: {stats['total_operations']} operations")
                return True
            
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_async_test())
            loop.close()
            
            self.assertTrue(result)
            
        except Exception as e:
            self.fail(f"Async Neural Pathways test failed: {e}")
    
    def test_003_work_stealing_thread_pool(self):
        """Test 3: Work-Stealing Thread Pool"""
        print("⚡ Testing Work-Stealing Thread Pool functionality...")
        
        try:
            from ai.reactive_neural_architecture import WorkStealingThreadPool
            
            # Create work-stealing pool
            pool = WorkStealingThreadPool(num_workers=4)
            
            # Test computation
            def cpu_intensive_task(n):
                # Simulate CPU-bound work
                result = 0
                for i in range(n * 1000):
                    result += i
                return result
            
            # Submit multiple tasks
            events = []
            for i in range(6):
                event = pool.submit(cpu_intensive_task, i + 1)
                events.append(event)
            
            # Wait for completion
            start_time = time.time()
            for event in events:
                event.wait(timeout=5.0)
            completion_time = time.time() - start_time
            
            # Get statistics
            stats = pool.get_stats()
            
            # Verify all tasks completed
            self.assertGreaterEqual(stats['tasks_completed'], 6)
            self.assertGreater(stats['tasks_submitted'], 0)
            
            # Should complete in reasonable time with parallelism
            self.assertLess(completion_time, 2.0)
            
            print(f"✅ Work-Stealing Pool test passed: {stats['tasks_completed']} tasks in {completion_time:.3f}s")
            
            # Cleanup
            pool.shutdown(wait=True)
            
        except Exception as e:
            self.fail(f"Work-Stealing Thread Pool test failed: {e}")
    
    def test_004_optimized_memory_manager(self):
        """Test 4: Optimized Memory Manager"""
        print("🧠 Testing Optimized Memory Manager functionality...")
        
        try:
            from ai.reactive_neural_architecture import OptimizedMemoryManager
            
            # Create memory manager
            memory_manager = OptimizedMemoryManager()
            
            # Test object pooling
            def create_test_object():
                return {'data': [1, 2, 3, 4, 5], 'timestamp': time.time()}
            
            def reset_test_object(obj):
                obj['data'] = []
                obj['timestamp'] = time.time()
            
            # Get objects from pool
            obj1 = memory_manager.get_pooled_object("test_pool", create_test_object)
            obj2 = memory_manager.get_pooled_object("test_pool", create_test_object)
            
            self.assertIsNotNone(obj1)
            self.assertIsNotNone(obj2)
            self.assertEqual(obj1['data'], [1, 2, 3, 4, 5])
            
            # Return to pool
            memory_manager.return_to_pool("test_pool", obj1, reset_test_object)
            
            # Get from pool again (should reuse)
            obj3 = memory_manager.get_pooled_object("test_pool")
            self.assertIsNotNone(obj3)
            self.assertEqual(obj3['data'], [])  # Should be reset
            
            # Test copy-on-write
            test_data = {'shared': 'data', 'numbers': [1, 2, 3]}
            cow_id = memory_manager.create_cow_object("test_cow", test_data)
            
            # Get reference (no copy)
            cow_obj1, cow_id1 = memory_manager.get_cow_object(cow_id, copy_on_write=False)
            self.assertEqual(cow_obj1['shared'], 'data')
            
            # Get copy for modification
            cow_obj2, cow_id2 = memory_manager.get_cow_object(cow_id, copy_on_write=True)
            cow_obj2['shared'] = 'modified'
            
            # Original should be unchanged
            self.assertEqual(cow_obj1['shared'], 'data')
            self.assertEqual(cow_obj2['shared'], 'modified')
            
            # Get statistics
            stats = memory_manager.get_memory_stats()
            self.assertGreater(stats['objects_pooled'], 0)
            self.assertGreater(stats['cow_objects'], 0)
            
            print(f"✅ Memory Manager test passed: {stats['objects_pooled']} pooled, {stats['cow_objects']} COW objects")
            
            # Cleanup
            memory_manager.release_cow_object(cow_id1)
            memory_manager.release_cow_object(cow_id2)
            
        except Exception as e:
            self.fail(f"Optimized Memory Manager test failed: {e}")
    
    def test_005_hybrid_consciousness_worker(self):
        """Test 5: Hybrid Consciousness Worker"""
        print("🤖 Testing Hybrid Consciousness Worker functionality...")
        
        try:
            from ai.reactive_neural_architecture import EventBus, AsyncNeuralPathways, WorkStealingThreadPool, OptimizedMemoryManager
            from ai.hybrid_consciousness_integration import HybridConsciousnessWorker, HybridWorkload, ProcessingMode
            from ai.parallel_processor import ParallelConsciousnessProcessor
            
            async def run_hybrid_test():
                # Create components
                event_bus = EventBus()
                await event_bus.initialize_async()
                
                async_pathways = AsyncNeuralPathways()
                work_stealing_pool = WorkStealingThreadPool(num_workers=2)
                memory_manager = OptimizedMemoryManager()
                parallel_processor = ParallelConsciousnessProcessor()
                
                # Create hybrid worker
                worker = HybridConsciousnessWorker(
                    worker_id="test_worker",
                    event_bus=event_bus,
                    async_pathways=async_pathways,
                    work_stealing_pool=work_stealing_pool,
                    memory_manager=memory_manager,
                    parallel_processor=parallel_processor
                )
                
                # Create test workload
                workload = HybridWorkload(
                    text="Hello, hybrid world!",
                    user="test_user",
                    processing_mode=ProcessingMode.HYBRID_BALANCED
                )
                
                # Process workload
                start_time = time.time()
                result = await worker.process_workload(workload)
                processing_time = time.time() - start_time
                
                # Verify result
                self.assertTrue(result.success)
                self.assertIsNotNone(result.result)
                self.assertEqual(result.workload_id, workload.id)
                self.assertLess(processing_time, 5.0)  # Should be fast
                
                print(f"✅ Hybrid Worker test passed: {processing_time:.3f}s, mode: {result.mode_used.name}")
                
                # Cleanup
                work_stealing_pool.shutdown(wait=True)
                await event_bus.shutdown()
                
                return True
            
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_hybrid_test())
            loop.close()
            
            self.assertTrue(result)
            
        except Exception as e:
            self.fail(f"Hybrid Consciousness Worker test failed: {e}")
    
    def test_006_reactive_integration_layer(self):
        """Test 6: Reactive Integration Layer"""
        print("🔗 Testing Reactive Integration Layer functionality...")
        
        try:
            from ai.hybrid_consciousness_integration import ReactiveIntegrationLayer, ProcessingMode
            
            async def run_integration_test():
                # Create integration layer
                integration_layer = ReactiveIntegrationLayer(num_workers=2)
                
                # Initialize
                await integration_layer.initialize()
                self.assertTrue(integration_layer.initialized)
                
                # Test user input processing
                start_time = time.time()
                result = await integration_layer.process_user_input(
                    text="Test reactive processing",
                    user="test_user",
                    processing_mode=ProcessingMode.HYBRID_BALANCED
                )
                processing_time = time.time() - start_time
                
                # Verify result
                self.assertTrue(result.success)
                self.assertIsNotNone(result.result)
                self.assertLess(processing_time, 10.0)  # Should be fast
                
                # Get system health
                health = integration_layer.get_system_health()
                self.assertIn('backpressure', health)
                self.assertIn('telemetry', health)
                self.assertIn('workers', health)
                
                print(f"✅ Integration Layer test passed: {processing_time:.3f}s")
                print(f"📊 System health: {health['backpressure']['state']}")
                
                # Shutdown
                await integration_layer.shutdown()
                
                return True
            
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_integration_test())
            loop.close()
            
            self.assertTrue(result)
            
        except Exception as e:
            self.fail(f"Reactive Integration Layer test failed: {e}")
    
    def test_007_backpressure_management(self):
        """Test 7: Backpressure Management"""
        print("📊 Testing Backpressure Management functionality...")
        
        try:
            from ai.hybrid_consciousness_integration import BackpressureManager, BackpressureState, HybridWorkload, ProcessingMode
            
            # Create backpressure manager
            bp_manager = BackpressureManager()
            
            # Wait for initial metrics
            time.sleep(2.0)
            
            # Get initial state
            initial_metrics = bp_manager.get_metrics()
            self.assertIn('state', initial_metrics)
            self.assertIn('metrics', initial_metrics)
            
            # Test throttling decision
            should_throttle = bp_manager.should_throttle()
            self.assertIsInstance(should_throttle, bool)
            
            # Test mode recommendation
            test_workload = HybridWorkload(
                text="Test workload",
                user="test_user",
                processing_mode=ProcessingMode.HYBRID_PERFORMANCE
            )
            
            recommended_mode = bp_manager.get_recommended_mode(test_workload)
            self.assertIsInstance(recommended_mode, ProcessingMode)
            
            # Update metrics manually
            bp_manager.update_metrics(
                response_times=0.5,
                event_queue_depth=50,
                async_operations=10
            )
            
            # Get updated metrics
            updated_metrics = bp_manager.get_metrics()
            self.assertGreaterEqual(len(updated_metrics['metrics']['response_times']), 1)
            
            print(f"✅ Backpressure Manager test passed: state={updated_metrics['state']}")
            
            # Cleanup
            bp_manager.monitoring_active = False
            
        except Exception as e:
            self.fail(f"Backpressure Management test failed: {e}")
    
    def test_008_telemetry_collection(self):
        """Test 8: Telemetry Collection"""
        print("📈 Testing Telemetry Collection functionality...")
        
        try:
            from ai.hybrid_consciousness_integration import TelemetryCollector
            
            # Create telemetry collector
            telemetry = TelemetryCollector()
            
            # Record various metrics
            telemetry.record_processing_time("test_component", "operation1", 0.1)
            telemetry.record_processing_time("test_component", "operation2", 0.2)
            telemetry.record_counter("test_counter", 5)
            telemetry.record_histogram("response_times", 0.15)
            telemetry.record_success("test_component")
            telemetry.record_error("test_component", "timeout")
            
            # Get metrics summary
            summary = telemetry.get_metrics_summary()
            
            # Verify metrics were recorded
            self.assertIn('processing_times', summary)
            self.assertIn('counters', summary)
            self.assertIn('histograms', summary)
            self.assertIn('error_rates', summary)
            
            # Check specific values
            op_key = "test_component.operation1"
            self.assertIn(op_key, summary['processing_times'])
            self.assertEqual(summary['processing_times'][op_key]['avg'], 0.1)
            
            self.assertEqual(summary['counters']['test_counter'], 5)
            
            print(f"✅ Telemetry Collection test passed: {len(summary['processing_times'])} metrics")
            
        except Exception as e:
            self.fail(f"Telemetry Collection test failed: {e}")
    
    def test_009_end_to_end_performance(self):
        """Test 9: End-to-End Performance Test"""
        print("🏁 Testing End-to-End Performance...")
        
        try:
            from ai.hybrid_consciousness_integration import ReactiveIntegrationLayer, ProcessingMode
            
            async def run_performance_test():
                # Create integration layer
                integration_layer = ReactiveIntegrationLayer(num_workers=4)
                await integration_layer.initialize()
                
                # Test multiple concurrent requests
                test_inputs = [
                    ("Hello world", "user1"),
                    ("How are you?", "user2"),
                    ("What's the weather?", "user3"),
                    ("Tell me a joke", "user4"),
                    ("Explain quantum physics", "user5")
                ]
                
                # Process all inputs concurrently
                start_time = time.time()
                
                tasks = []
                for text, user in test_inputs:
                    task = integration_layer.process_user_input(
                        text=text,
                        user=user,
                        processing_mode=ProcessingMode.HYBRID_PERFORMANCE
                    )
                    tasks.append(task)
                
                # Wait for all to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                total_time = time.time() - start_time
                
                # Analyze results
                successful_results = [r for r in results if hasattr(r, 'success') and r.success]
                failed_results = [r for r in results if isinstance(r, Exception)]
                
                success_rate = len(successful_results) / len(test_inputs) * 100
                avg_processing_time = sum(r.processing_time for r in successful_results) / len(successful_results) if successful_results else 0
                
                # Performance targets
                self.assertGreater(success_rate, 80)  # At least 80% success rate
                self.assertLess(total_time, 30.0)  # Total time under 30 seconds
                self.assertLess(avg_processing_time, 10.0)  # Average under 10 seconds per request
                
                print(f"✅ Performance test passed:")
                print(f"   📊 Success rate: {success_rate:.1f}%")
                print(f"   ⏱️  Total time: {total_time:.3f}s")
                print(f"   📈 Avg per request: {avg_processing_time:.3f}s")
                print(f"   🚀 Throughput: {len(test_inputs)/total_time:.2f} requests/second")
                
                # Get final system health
                health = integration_layer.get_system_health()
                print(f"   🏥 System state: {health['backpressure']['state']}")
                
                await integration_layer.shutdown()
                return True
            
            # Run performance test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_performance_test())
            loop.close()
            
            self.assertTrue(result)
            
        except Exception as e:
            self.fail(f"End-to-End Performance test failed: {e}")

def run_reactive_architecture_tests():
    """Run all reactive neural architecture tests"""
    print("\n" + "="*80)
    print("🧪 REACTIVE NEURAL ARCHITECTURE COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestReactiveNeuralArchitecture)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=None)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  • {test}: {traceback}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  • {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED! Reactive Neural Architecture is fully functional.")
    else:
        print(f"\n⚠️  Some tests failed. Please review the failures above.")
    
    print("="*80)
    return result.wasSuccessful()

if __name__ == "__main__":
    # Set up logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(name)s: %(message)s'
    )
    
    success = run_reactive_architecture_tests()
    exit(0 if success else 1)