#!/usr/bin/env python3
"""
Buddy QA Agent - Launches Buddy and analyzes real-time logs
Created to detect every single issue in Buddy's operation
"""

import subprocess
import json
import time
import threading
import os
import signal
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

class BuddyQAAgent:
    """
    Quality Assurance agent that monitors Buddy's operation and detects issues
    """
    
    def __init__(self, buddy_script: str = "main.py"):
        self.buddy_script = buddy_script
        self.buddy_process = None
        self.events_file = "buddy_events.json"
        self.session_logs_file = "session_logs.json"
        self.report_file = "buddy_qa_report.txt"
        
        self.monitoring = False
        self.events_history = []
        self.analysis_results = {}
        
        # Issue detection counters
        self.issues_detected = {
            "double_speaking": 0,
            "whisper_failures": 0,
            "kokoro_overlaps": 0,
            "koboldcpp_stalls": 0,
            "memory_update_failures": 0,
            "runtime_errors": 0,
            "infinite_loops": 0,
            "response_timeouts": 0,
            "tts_queue_issues": 0,
            "vad_failures": 0
        }
        
        print("[BuddyQA] 🔍 Quality Assurance Agent initialized")
        print(f"[BuddyQA] 📁 Monitoring: {self.events_file}")
        print(f"[BuddyQA] 📊 Will launch: {self.buddy_script}")
    
    def launch_buddy(self) -> bool:
        """
        Launch Buddy as a subprocess and start monitoring
        """
        try:
            print(f"[BuddyQA] 🚀 Launching {self.buddy_script}...")
            
            # Clear any existing event logs
            if os.path.exists(self.events_file):
                os.remove(self.events_file)
                print(f"[BuddyQA] 🧹 Cleared old events file")
            
            # Start Buddy process
            self.buddy_process = subprocess.Popen(
                [sys.executable, self.buddy_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            print(f"[BuddyQA] ✅ Buddy launched with PID: {self.buddy_process.pid}")
            return True
            
        except Exception as e:
            print(f"[BuddyQA] ❌ Failed to launch Buddy: {e}")
            return False
    
    def start_monitoring(self):
        """
        Start monitoring Buddy's logs and events
        """
        self.monitoring = True
        
        # Start threads for different monitoring tasks
        stdout_thread = threading.Thread(target=self._monitor_stdout, daemon=True)
        stderr_thread = threading.Thread(target=self._monitor_stderr, daemon=True)
        events_thread = threading.Thread(target=self._monitor_events_file, daemon=True)
        analysis_thread = threading.Thread(target=self._continuous_analysis, daemon=True)
        
        stdout_thread.start()
        stderr_thread.start()
        events_thread.start()
        analysis_thread.start()
        
        print("[BuddyQA] 📊 Started monitoring threads")
        print("[BuddyQA] 🔍 Watching for issues...")
        
        return stdout_thread, stderr_thread, events_thread, analysis_thread
    
    def _monitor_stdout(self):
        """Monitor Buddy's stdout for important messages"""
        if not self.buddy_process:
            return
        
        while self.monitoring and self.buddy_process.poll() is None:
            try:
                line = self.buddy_process.stdout.readline()
                if line:
                    print(f"[Buddy] {line.strip()}")
                    self._analyze_stdout_line(line.strip())
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"[BuddyQA] ⚠️ stdout monitoring error: {e}")
                break
    
    def _monitor_stderr(self):
        """Monitor Buddy's stderr for errors"""
        if not self.buddy_process:
            return
        
        while self.monitoring and self.buddy_process.poll() is None:
            try:
                line = self.buddy_process.stderr.readline()
                if line:
                    print(f"[Buddy ERROR] {line.strip()}")
                    self._analyze_stderr_line(line.strip())
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"[BuddyQA] ⚠️ stderr monitoring error: {e}")
                break
    
    def _monitor_events_file(self):
        """Monitor the buddy_events.json file for new events"""
        last_event_count = 0
        
        while self.monitoring:
            try:
                if os.path.exists(self.events_file):
                    with open(self.events_file, 'r') as f:
                        events = json.load(f)
                    
                    # Process new events
                    if len(events) > last_event_count:
                        new_events = events[last_event_count:]
                        for event in new_events:
                            self._analyze_event(event)
                        last_event_count = len(events)
                        self.events_history = events
                
                time.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                print(f"[BuddyQA] ⚠️ Events monitoring error: {e}")
                time.sleep(1)
    
    def _continuous_analysis(self):
        """Continuously analyze events for patterns and issues"""
        while self.monitoring:
            try:
                self._detect_real_time_issues()
                time.sleep(2)  # Analyze every 2 seconds
            except Exception as e:
                print(f"[BuddyQA] ⚠️ Analysis error: {e}")
                time.sleep(5)
    
    def _analyze_stdout_line(self, line: str):
        """Analyze individual stdout lines for issues"""
        # Detect potential infinite loops
        if "Describe this experience" in line and "feeling_" in line:
            self.issues_detected["infinite_loops"] += 1
            print(f"[BuddyQA] 🔄 Potential infinite loop detected: {line[:100]}...")
        
        # Detect double speaking issues
        if "Speaking chunk" in line:
            # Check if multiple chunks are being spoken too quickly
            current_time = time.time()
            if not hasattr(self, '_last_chunk_time'):
                self._last_chunk_time = current_time
                self._chunk_count_in_second = 0
            
            if current_time - self._last_chunk_time < 0.1:
                self._chunk_count_in_second += 1
                if self._chunk_count_in_second > 10:
                    self.issues_detected["double_speaking"] += 1
                    print(f"[BuddyQA] 🗣️ Double speaking detected: {self._chunk_count_in_second} chunks in 0.1s")
            else:
                self._chunk_count_in_second = 0
                self._last_chunk_time = current_time
    
    def _analyze_stderr_line(self, line: str):
        """Analyze stderr lines for errors"""
        if any(error_word in line.lower() for error_word in ["error", "exception", "traceback", "failed"]):
            self.issues_detected["runtime_errors"] += 1
            print(f"[BuddyQA] ❌ Runtime error detected: {line[:100]}...")
    
    def _analyze_event(self, event: Dict[str, Any]):
        """Analyze individual events for issues"""
        event_type = event.get("event_type", "")
        component = event.get("component", "")
        data = event.get("data", {})
        
        # Detect Whisper STT failures
        if event_type == "stt_finish" and not data.get("success", True):
            self.issues_detected["whisper_failures"] += 1
            print(f"[BuddyQA] 🎤 Whisper STT failure: {data.get('error', 'Unknown error')}")
        
        # Detect long STT latency
        if event_type == "stt_finish" and data.get("latency", 0) > 5.0:
            print(f"[BuddyQA] ⏱️ Slow STT: {data.get('latency', 0):.2f}s")
        
        # Detect TTS issues
        if event_type == "tts_finish" and not data.get("success", True):
            self.issues_detected["kokoro_overlaps"] += 1
            print(f"[BuddyQA] 🔊 TTS failure: {data.get('error', 'Unknown error')}")
        
        # Detect LLM stalls
        if event_type == "llm_finish" and data.get("latency", 0) > 10.0:
            self.issues_detected["koboldcpp_stalls"] += 1
            print(f"[BuddyQA] 🧠 LLM stall: {data.get('latency', 0):.2f}s")
        
        # Detect memory update issues
        if event_type == "memory_update" and not data.get("success", True):
            self.issues_detected["memory_update_failures"] += 1
            print(f"[BuddyQA] 💾 Memory update failure: {data.get('error', 'Unknown error')}")
        
        # Detect TTS queue issues
        if event_type == "tts_queue" and data.get("action") == "skipped":
            self.issues_detected["tts_queue_issues"] += 1
            print(f"[BuddyQA] ⏩ TTS chunk skipped: {data.get('chunk_id', 'unknown')}")
    
    def _detect_real_time_issues(self):
        """Detect issues from patterns in recent events"""
        if len(self.events_history) < 2:
            return
        
        recent_events = self.events_history[-50:]  # Last 50 events
        
        # Detect Kokoro overlapping playback
        tts_starts = [e for e in recent_events if e.get("event_type") == "tts_playback_start"]
        tts_finishes = [e for e in recent_events if e.get("event_type") == "tts_playback_finish"]
        
        if len(tts_starts) > len(tts_finishes) + 2:
            self.issues_detected["kokoro_overlaps"] += 1
            print(f"[BuddyQA] 🔊 TTS overlap detected: {len(tts_starts)} starts vs {len(tts_finishes)} finishes")
        
        # Detect response timeouts
        conversation_starts = [e for e in recent_events if e.get("event_type") == "conversation_start"]
        llm_starts = [e for e in recent_events if e.get("event_type") == "llm_start"]
        
        if conversation_starts and not llm_starts:
            # Conversation started but no LLM response
            last_conversation = conversation_starts[-1]
            conversation_time = datetime.fromisoformat(last_conversation["timestamp"])
            if (datetime.now() - conversation_time).total_seconds() > 15:
                self.issues_detected["response_timeouts"] += 1
                print(f"[BuddyQA] ⏱️ Response timeout detected: >15s since conversation start")
    
    def stop_monitoring(self):
        """Stop monitoring and terminate Buddy"""
        print("[BuddyQA] 🛑 Stopping monitoring...")
        self.monitoring = False
        
        if self.buddy_process and self.buddy_process.poll() is None:
            print("[BuddyQA] 🔒 Terminating Buddy process...")
            try:
                self.buddy_process.terminate()
                self.buddy_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("[BuddyQA] ⚡ Force killing Buddy process...")
                self.buddy_process.kill()
                self.buddy_process.wait()
            
            print(f"[BuddyQA] ✅ Buddy process terminated")
    
    def generate_final_report(self):
        """Generate comprehensive analysis report"""
        print("[BuddyQA] 📊 Generating final analysis report...")
        
        # Calculate session duration
        if self.events_history:
            session_start = datetime.fromisoformat(self.events_history[0]["timestamp"])
            session_end = datetime.now()
            session_duration = (session_end - session_start).total_seconds()
        else:
            session_duration = 0
        
        # Analyze event patterns
        event_counts = {}
        component_usage = {}
        error_events = []
        
        for event in self.events_history:
            event_type = event.get("event_type", "unknown")
            component = event.get("component", "unknown")
            
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            component_usage[component] = component_usage.get(component, 0) + 1
            
            if event_type in ["error", "warning"]:
                error_events.append(event)
        
        # Calculate performance metrics
        stt_events = [e for e in self.events_history if e.get("event_type") == "stt_finish"]
        tts_events = [e for e in self.events_history if e.get("event_type") == "tts_finish"]
        llm_events = [e for e in self.events_history if e.get("event_type") == "llm_finish"]
        
        avg_stt_latency = sum(e.get("data", {}).get("latency", 0) for e in stt_events) / len(stt_events) if stt_events else 0
        avg_tts_latency = sum(e.get("data", {}).get("latency", 0) for e in tts_events) / len(tts_events) if tts_events else 0
        avg_llm_latency = sum(e.get("data", {}).get("latency", 0) for e in llm_events) / len(llm_events) if llm_events else 0
        
        # Generate report
        report = self._create_report_content(
            session_duration, event_counts, component_usage, 
            error_events, avg_stt_latency, avg_tts_latency, avg_llm_latency
        )
        
        # Save reports
        self._save_session_logs()
        self._save_report(report)
        
        print(f"[BuddyQA] ✅ Reports saved:")
        print(f"[BuddyQA]   📄 {self.report_file}")
        print(f"[BuddyQA]   📊 {self.session_logs_file}")
        
        return report
    
    def _create_report_content(self, session_duration, event_counts, component_usage, 
                              error_events, avg_stt_latency, avg_tts_latency, avg_llm_latency):
        """Create the report content"""
        report = []
        report.append("=" * 60)
        report.append("BUDDY QA ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Session Duration: {session_duration:.1f} seconds")
        report.append(f"Total Events Logged: {len(self.events_history)}")
        report.append("")
        
        # Issues Summary
        report.append("ISSUES DETECTED:")
        report.append("-" * 30)
        total_issues = sum(self.issues_detected.values())
        report.append(f"Total Issues: {total_issues}")
        for issue_type, count in self.issues_detected.items():
            if count > 0:
                report.append(f"  ❌ {issue_type.replace('_', ' ').title()}: {count}")
        report.append("")
        
        # Performance Breakdown
        report.append("LATENCY BREAKDOWN:")
        report.append("-" * 30)
        report.append(f"Average STT (Whisper) Latency: {avg_stt_latency:.3f}s")
        report.append(f"Average TTS (Kokoro) Latency: {avg_tts_latency:.3f}s") 
        report.append(f"Average LLM (KoboldCPP) Latency: {avg_llm_latency:.3f}s")
        report.append("")
        
        # Component Usage
        report.append("COMPONENT USAGE:")
        report.append("-" * 30)
        for component, count in sorted(component_usage.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {component}: {count} events")
        report.append("")
        
        # Event Type Breakdown
        report.append("EVENT TYPE BREAKDOWN:")
        report.append("-" * 30)
        for event_type, count in sorted(event_counts.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {event_type}: {count}")
        report.append("")
        
        # Runtime Errors
        if error_events:
            report.append("RUNTIME ERRORS:")
            report.append("-" * 30)
            for i, error in enumerate(error_events[-10:], 1):  # Last 10 errors
                timestamp = error.get("timestamp", "unknown")
                component = error.get("component", "unknown")
                data = error.get("data", {})
                message = data.get("error_message", data.get("warning_message", "Unknown error"))
                report.append(f"  {i}. [{timestamp[:19]}] {component}: {message}")
            report.append("")
        
        # Optimization Suggestions
        report.append("OPTIMIZATION SUGGESTIONS:")
        report.append("-" * 30)
        suggestions = self._generate_optimization_suggestions()
        for suggestion in suggestions:
            report.append(f"  • {suggestion}")
        
        return "\n".join(report)
    
    def _generate_optimization_suggestions(self):
        """Generate optimization suggestions based on detected issues"""
        suggestions = []
        
        if self.issues_detected["double_speaking"] > 0:
            suggestions.append("Fix double speaking: Add proper TTS queue management and chunk spacing")
        
        if self.issues_detected["whisper_failures"] > 0:
            suggestions.append("Improve STT reliability: Check microphone input quality and Whisper model parameters")
        
        if self.issues_detected["kokoro_overlaps"] > 0:
            suggestions.append("Fix TTS overlaps: Implement proper audio playback synchronization")
        
        if self.issues_detected["koboldcpp_stalls"] > 0:
            suggestions.append("Optimize LLM performance: Check KoboldCPP configuration and network connectivity")
        
        if self.issues_detected["memory_update_failures"] > 0:
            suggestions.append("Fix memory system: Check file permissions and memory module initialization")
        
        if self.issues_detected["infinite_loops"] > 0:
            suggestions.append("Critical: Fix consciousness loop preventing wake word detection")
        
        if self.issues_detected["response_timeouts"] > 0:
            suggestions.append("Reduce response timeouts: Optimize LLM processing pipeline")
        
        if not suggestions:
            suggestions.append("No major issues detected - system appears to be functioning well!")
        
        return suggestions
    
    def _save_session_logs(self):
        """Save detailed session logs"""
        session_data = {
            "session_info": {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": time.time() - (self.events_history[0].get("unix_time", time.time()) if self.events_history else time.time()),
                "total_events": len(self.events_history),
                "buddy_script": self.buddy_script
            },
            "issues_detected": self.issues_detected,
            "events": self.events_history
        }
        
        with open(self.session_logs_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
    
    def _save_report(self, report_content: str):
        """Save the readable report"""
        with open(self.report_file, 'w') as f:
            f.write(report_content)
    
    def run_qa_session(self, duration_minutes: int = 10):
        """
        Run a complete QA session
        
        Args:
            duration_minutes: How long to monitor Buddy (default: 10 minutes)
        """
        print(f"[BuddyQA] 🚀 Starting {duration_minutes}-minute QA session...")
        
        try:
            # Launch Buddy
            if not self.launch_buddy():
                return False
            
            # Start monitoring
            threads = self.start_monitoring()
            
            # Wait for specified duration
            print(f"[BuddyQA] ⏱️ Monitoring for {duration_minutes} minutes...")
            print("[BuddyQA] 💡 Interact with Buddy normally - the QA agent is watching!")
            print("[BuddyQA] 🛑 Press Ctrl+C to stop early")
            
            try:
                time.sleep(duration_minutes * 60)
            except KeyboardInterrupt:
                print("\n[BuddyQA] ⏹️ QA session stopped by user")
            
            # Generate final report
            self.stop_monitoring()
            report = self.generate_final_report()
            
            print("\n[BuddyQA] ✅ QA session complete!")
            print("\nQUICK SUMMARY:")
            print("-" * 40)
            total_issues = sum(self.issues_detected.values())
            if total_issues == 0:
                print("🎉 No issues detected! Buddy is running smoothly.")
            else:
                print(f"⚠️ {total_issues} issues detected:")
                for issue_type, count in self.issues_detected.items():
                    if count > 0:
                        print(f"   • {issue_type.replace('_', ' ').title()}: {count}")
            
            return True
            
        except Exception as e:
            print(f"[BuddyQA] ❌ QA session failed: {e}")
            return False

def main():
    """Main entry point for the QA agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Buddy QA Agent - Monitor and analyze Buddy's operation")
    parser.add_argument("--duration", "-d", type=int, default=10, help="Monitoring duration in minutes (default: 10)")
    parser.add_argument("--buddy-script", "-s", type=str, default="main.py", help="Buddy script to launch (default: main.py)")
    parser.add_argument("--continuous", "-c", action="store_true", help="Run continuously until stopped")
    
    args = parser.parse_args()
    
    qa_agent = BuddyQAAgent(args.buddy_script)
    
    if args.continuous:
        print("[BuddyQA] 🔄 Running in continuous mode - press Ctrl+C to stop")
        duration = 999999  # Very long duration
    else:
        duration = args.duration
    
    qa_agent.run_qa_session(duration)

if __name__ == "__main__":
    main()