#!/usr/bin/env python3
"""
Test QA Agent - Quick test without launching main.py
"""

import json
import time
from buddy_qa_agent import BuddyQAAgent

def test_qa_agent_analysis():
    print("Testing QA Agent analysis capabilities...")
    
    # Create a QA agent
    qa_agent = BuddyQAAgent()
    
    # Load the test events we just created
    qa_agent.events_history = []
    if 'buddy_events.json' in locals() or True:
        try:
            with open('buddy_events.json', 'r') as f:
                qa_agent.events_history = json.load(f)
        except:
            print("No test events found, creating some...")
    
    # Test analysis on the events
    print(f"Analyzing {len(qa_agent.events_history)} events...")
    
    # Simulate some issues for testing
    qa_agent.issues_detected["runtime_errors"] = 1  # From our test error
    qa_agent.issues_detected["memory_update_failures"] = 0
    
    # Run analysis
    qa_agent._detect_real_time_issues()
    
    # Generate report
    report = qa_agent.generate_final_report()
    
    print("\n" + "="*50)
    print("SAMPLE QA REPORT:")
    print("="*50)
    print(report[:1000] + "..." if len(report) > 1000 else report)
    
    print("\n✅ QA Agent test complete!")
    print("📄 Full report saved to buddy_qa_report.txt")

if __name__ == "__main__":
    test_qa_agent_analysis()