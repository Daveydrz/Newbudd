#!/usr/bin/env python3
"""
Class 5 Consciousness Integration Audit Script

This script verifies if Buddy is properly using Class 5 consciousness in the main prompt
and validates the integration of Memory, Mood, Goals, Thoughts, and Personality components.

Created for: Automated Codebase Audit: Buddy Class 5 Consciousness Integration
"""

import os
import sys
import re
import json
import ast
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

class Class5ConsciousnessAuditor:
    """Comprehensive auditor for Class 5 consciousness integration"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.ai_path = self.repo_path / "ai"
        self.results = {
            "audit_timestamp": datetime.now().isoformat(),
            "repository_path": str(repo_path),
            "class5_verification": {},
            "prompt_builder_analysis": {},
            "consciousness_components": {},
            "integration_status": {},
            "issues_found": [],
            "recommendations": []
        }
        
    def run_complete_audit(self) -> Dict[str, Any]:
        """Run complete Class 5 consciousness audit"""
        print("🧠 Starting Class 5 Consciousness Integration Audit...")
        print("=" * 60)
        
        # 1. Verify Class 5 consciousness usage in main LLM functions
        print("\n1️⃣ Verifying Class 5 Consciousness in Main LLM Functions...")
        self._audit_llm_functions()
        
        # 2. Confirm main prompt usage of consciousness components
        print("\n2️⃣ Confirming Main Prompt Usage of Consciousness Components...")
        self._audit_prompt_consciousness_usage()
        
        # 3. Audit get_consciousness_snapshot() function
        print("\n3️⃣ Auditing get_consciousness_snapshot() Function...")
        self._audit_consciousness_snapshot()
        
        # 4. Detect duplicate prompt builders
        print("\n4️⃣ Detecting Duplicate Prompt Builders...")
        self._audit_prompt_builders()
        
        # 5. Verify consciousness component completeness
        print("\n5️⃣ Verifying Consciousness Component Completeness...")
        self._audit_consciousness_components()
        
        # 6. Generate final analysis
        print("\n6️⃣ Generating Final Analysis...")
        self._generate_final_analysis()
        
        print("\n✅ Audit Complete!")
        return self.results
    
    def _audit_llm_functions(self):
        """Audit main LLM functions for Class 5 consciousness usage"""
        llm_functions = [
            ("ai/llm_handler.py", ["generate_response", "call_llm", "generate_response_with_consciousness"]),
            ("ai/chat.py", ["generate_response", "generate_response_streaming"]),
            ("ai/chat_enhanced.py", ["generate_response_with_human_memory"]),
            ("ai/chat_enhanced_smart.py", ["generate_response_streaming_with_smart_memory"]),
            ("ai/chat_enhanced_smart_with_fusion.py", ["generate_response_streaming_with_intelligent_fusion"]),
            ("main.py", ["generate_response"])
        ]
        
        consciousness_usage = {}
        
        for file_path, functions in llm_functions:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                continue
                
            print(f"   📁 Analyzing {file_path}...")
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_analysis = {
                    "file_exists": True,
                    "functions_found": [],
                    "class5_imports": [],
                    "consciousness_calls": [],
                    "uses_class5": False,
                    "alternative_prompt_builders": [],
                    "direct_llm_calls": []
                }
                
                # Check for function definitions
                for func_name in functions:
                    if re.search(rf'def\s+{func_name}\s*\(', content):
                        file_analysis["functions_found"].append(func_name)
                
                # Check for Class 5 consciousness imports
                class5_patterns = [
                    r'from\s+.*class5_consciousness_integration',
                    r'import\s+.*class5_consciousness_integration',
                    r'from\s+.*conscious_prompt_builder.*import.*get_consciousness_snapshot',
                    r'consciousness_integrated_response',
                    r'get_consciousness_snapshot',
                    r'build_consciousness_integrated_prompt'
                ]
                
                for pattern in class5_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        file_analysis["class5_imports"].extend(matches)
                        file_analysis["uses_class5"] = True
                
                # Check for consciousness function calls
                consciousness_calls = [
                    r'get_consciousness_snapshot\s*\(',
                    r'build_consciousness_integrated_prompt\s*\(',
                    r'consciousness_integrated_response\s*\(',
                    r'process_user_input_with_consciousness\s*\(',
                    r'generate_consciousness_integrated_response\s*\(',
                    r'generate_response_with_consciousness\s*\('
                ]
                
                for pattern in consciousness_calls:
                    matches = re.findall(pattern, content)
                    if matches:
                        file_analysis["consciousness_calls"].extend(matches)
                        file_analysis["uses_class5"] = True
                
                # Check for alternative prompt builders used instead
                alt_prompt_patterns = [
                    r'compress_prompt\s*\(',
                    r'expand_prompt\s*\(',
                    r'build_optimized_prompt\s*\(',
                    r'_build_enhanced_prompt\s*\(',
                    r'from\s+ai\.prompt_compressor\s+import',
                    r'from\s+ai\.optimized_prompt_builder\s+import'
                ]
                
                for pattern in alt_prompt_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        file_analysis["alternative_prompt_builders"].extend(matches)
                
                # Check for direct LLM calls (non-consciousness)
                direct_llm_patterns = [
                    r'ask_kobold\s*\(',
                    r'ask_kobold_streaming\s*\(',
                    r'requests\.post\s*\(',
                    r'generate_response_streaming\s*\('
                ]
                
                for pattern in direct_llm_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        file_analysis["direct_llm_calls"].extend(matches)
                
                consciousness_usage[file_path] = file_analysis
                
                # Print findings
                if file_analysis["uses_class5"]:
                    print(f"      ✅ Class 5 consciousness detected")
                    print(f"         Functions: {file_analysis['functions_found']}")
                    print(f"         Imports: {len(file_analysis['class5_imports'])}")
                    print(f"         Calls: {len(file_analysis['consciousness_calls'])}")
                else:
                    print(f"      ❌ No Class 5 consciousness usage found")
                    if file_analysis["functions_found"]:
                        print(f"         Functions found: {file_analysis['functions_found']}")
                        print(f"         Alternative builders: {len(file_analysis['alternative_prompt_builders'])}")
                        print(f"         Direct LLM calls: {len(file_analysis['direct_llm_calls'])}")
                        self.results["issues_found"].append(
                            f"LLM functions in {file_path} do not use Class 5 consciousness integration"
                        )
                
            except Exception as e:
                print(f"      ❌ Error analyzing {file_path}: {e}")
                consciousness_usage[file_path] = {"error": str(e)}
        
        self.results["class5_verification"] = consciousness_usage
    
    def _audit_prompt_consciousness_usage(self):
        """Audit prompt building logic for consciousness component usage"""
        prompt_files = [
            "ai/conscious_prompt_builder.py",
            "ai/optimized_prompt_builder.py", 
            "ai/llm_handler.py"
        ]
        
        consciousness_components = ["Memory", "Mood", "Goals", "Thoughts", "Personality"]
        prompt_analysis = {}
        
        for file_path in prompt_files:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                continue
                
            print(f"   📁 Analyzing prompt logic in {file_path}...")
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_analysis = {
                    "components_detected": {},
                    "prompt_build_functions": [],
                    "consciousness_integration": False
                }
                
                # Check for consciousness components
                component_patterns = {
                    "Memory": [r'memory_timeline', r'memory_context', r'recent_memories', r'MemoryType'],
                    "Mood": [r'mood_manager', r'emotional_state', r'mood_modifiers', r'MoodState'],
                    "Goals": [r'goal_manager', r'active_goals', r'goal_progress', r'GoalType'],
                    "Thoughts": [r'thought_loop', r'inner_thoughts', r'thought_summary', r'ThoughtType'],
                    "Personality": [r'personality_profile', r'personality_modifiers', r'interaction_style', r'PersonalityDimension']
                }
                
                for component, patterns in component_patterns.items():
                    component_found = False
                    matches = []
                    for pattern in patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            component_found = True
                            matches.extend(re.findall(pattern, content, re.IGNORECASE))
                    
                    file_analysis["components_detected"][component] = {
                        "found": component_found,
                        "matches": len(matches),
                        "references": matches[:5]  # First 5 matches
                    }
                
                # Check for prompt building functions
                prompt_functions = [
                    r'def\s+.*build.*prompt.*\(',
                    r'def\s+.*consciousness.*prompt.*\(',
                    r'def\s+.*get_consciousness_snapshot.*\(',
                    r'def\s+.*build_consciousness_integrated_prompt.*\('
                ]
                
                for pattern in prompt_functions:
                    matches = re.findall(pattern, content)
                    file_analysis["prompt_build_functions"].extend(matches)
                
                # Determine if consciousness integration is present
                components_found = sum(1 for comp in file_analysis["components_detected"].values() if comp["found"])
                file_analysis["consciousness_integration"] = components_found >= 3
                
                prompt_analysis[file_path] = file_analysis
                
                print(f"      Components found: {components_found}/{len(consciousness_components)}")
                print(f"      Prompt functions: {len(file_analysis['prompt_build_functions'])}")
                print(f"      Integration status: {'✅ Good' if file_analysis['consciousness_integration'] else '❌ Insufficient'}")
                
                if not file_analysis["consciousness_integration"]:
                    self.results["issues_found"].append(
                        f"Insufficient consciousness integration in {file_path} ({components_found}/{len(consciousness_components)} components)"
                    )
                
            except Exception as e:
                print(f"      ❌ Error analyzing {file_path}: {e}")
                prompt_analysis[file_path] = {"error": str(e)}
        
        self.results["prompt_builder_analysis"] = prompt_analysis
    
    def _audit_consciousness_snapshot(self):
        """Audit the get_consciousness_snapshot() function"""
        snapshot_locations = []
        
        # Search for get_consciousness_snapshot function
        for py_file in self.ai_path.glob("**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if re.search(r'def\s+get_consciousness_snapshot\s*\(', content):
                    snapshot_locations.append(str(py_file.relative_to(self.repo_path)))
                    
            except Exception as e:
                print(f"   ⚠️ Error reading {py_file}: {e}")
        
        print(f"   📍 Found get_consciousness_snapshot() in {len(snapshot_locations)} files:")
        
        snapshot_analysis = {
            "locations": snapshot_locations,
            "function_analysis": {}
        }
        
        # Analyze each implementation
        for location in snapshot_locations:
            full_path = self.repo_path / location
            print(f"      📁 Analyzing {location}...")
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find the function and analyze its implementation
                func_match = re.search(
                    r'def\s+get_consciousness_snapshot\s*\([^)]*\).*?(?=\ndef|\nclass|\Z)',
                    content, 
                    re.DOTALL
                )
                
                if func_match:
                    func_body = func_match.group(0)
                    
                    analysis = {
                        "memory_integration": bool(re.search(r'memory|Memory', func_body)),
                        "mood_integration": bool(re.search(r'mood|Mood', func_body)),
                        "goals_integration": bool(re.search(r'goal|Goal', func_body)),
                        "thoughts_integration": bool(re.search(r'thought|inner_monologue', func_body)),
                        "personality_integration": bool(re.search(r'personality|Personality', func_body)),
                        "function_length": len(func_body.split('\n')),
                        "returns_comprehensive_state": bool(re.search(r'return.*snapshot|return.*state', func_body))
                    }
                    
                    components_integrated = sum([
                        analysis["memory_integration"],
                        analysis["mood_integration"], 
                        analysis["goals_integration"],
                        analysis["thoughts_integration"],
                        analysis["personality_integration"]
                    ])
                    
                    analysis["completeness_score"] = components_integrated / 5.0
                    analysis["is_comprehensive"] = components_integrated >= 4
                    
                    snapshot_analysis["function_analysis"][location] = analysis
                    
                    print(f"         Components: {components_integrated}/5")
                    print(f"         Comprehensive: {'✅ Yes' if analysis['is_comprehensive'] else '❌ No'}")
                    
                    if not analysis["is_comprehensive"]:
                        self.results["issues_found"].append(
                            f"get_consciousness_snapshot() in {location} is missing consciousness components ({components_integrated}/5)"
                        )
                
            except Exception as e:
                print(f"         ❌ Error analyzing function: {e}")
                snapshot_analysis["function_analysis"][location] = {"error": str(e)}
        
        if not snapshot_locations:
            print("   ❌ get_consciousness_snapshot() function not found!")
            self.results["issues_found"].append("get_consciousness_snapshot() function not found in codebase")
        
        self.results["consciousness_components"]["snapshot_function"] = snapshot_analysis
    
    def _audit_prompt_builders(self):
        """Detect and analyze duplicate prompt builders"""
        prompt_builder_files = []
        
        # Search for prompt builder files
        search_patterns = [
            "**/*prompt*builder*.py",
            "**/chat*.py",
            "**/llm*.py"
        ]
        
        for pattern in search_patterns:
            for py_file in self.repo_path.glob(pattern):
                if py_file.name not in [f.name for f in prompt_builder_files]:
                    prompt_builder_files.append(py_file)
        
        print(f"   📍 Found {len(prompt_builder_files)} potential prompt builder files:")
        
        builder_analysis = {
            "total_files": len(prompt_builder_files),
            "file_analysis": {},
            "build_prompt_functions": [],
            "main_usage": None,
            "duplicates_detected": False
        }
        
        all_build_functions = []
        
        for py_file in prompt_builder_files:
            relative_path = str(py_file.relative_to(self.repo_path))
            print(f"      📁 {relative_path}")
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find build_prompt and similar functions
                build_functions = re.findall(
                    r'def\s+(.*build.*prompt.*|.*prompt.*build.*)\s*\(',
                    content,
                    re.IGNORECASE
                )
                
                consciousness_aware = bool(re.search(r'consciousness|Consciousness', content))
                
                file_info = {
                    "path": relative_path,
                    "build_functions": build_functions,
                    "consciousness_aware": consciousness_aware,
                    "function_count": len(build_functions)
                }
                
                builder_analysis["file_analysis"][relative_path] = file_info
                all_build_functions.extend([(func, relative_path) for func in build_functions])
                
                print(f"         Functions: {len(build_functions)}")
                print(f"         Consciousness-aware: {'✅ Yes' if consciousness_aware else '❌ No'}")
                
            except Exception as e:
                print(f"         ❌ Error analyzing: {e}")
                builder_analysis["file_analysis"][relative_path] = {"error": str(e)}
        
        builder_analysis["build_prompt_functions"] = all_build_functions
        
        # Check for duplicates
        function_names = [func[0] for func in all_build_functions]
        if len(function_names) != len(set(function_names)):
            builder_analysis["duplicates_detected"] = True
            print("   ⚠️ Duplicate function names detected!")
            self.results["issues_found"].append("Duplicate prompt building functions detected")
        
        # Determine which is used by main generate_response
        self._find_main_prompt_builder_usage(builder_analysis)
        
        self.results["prompt_builder_analysis"]["builder_detection"] = builder_analysis
    
    def _find_main_prompt_builder_usage(self, builder_analysis):
        """Find which prompt builder is used by main generate_response functions"""
        main_files = ["main.py", "ai/llm_handler.py"]
        
        print("   🔍 Tracing main prompt builder usage...")
        
        for main_file in main_files:
            full_path = self.repo_path / main_file
            if not full_path.exists():
                continue
                
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for imports and function calls
                imports = re.findall(r'from\s+[\w.]+\s+import\s+.*prompt.*', content, re.IGNORECASE)
                calls = re.findall(r'(\w*prompt\w*)\s*\(', content, re.IGNORECASE)
                
                print(f"      📁 {main_file}:")
                print(f"         Prompt imports: {len(imports)}")
                print(f"         Prompt calls: {len(set(calls))}")
                
                if imports or calls:
                    builder_analysis["main_usage"] = {
                        "file": main_file,
                        "imports": imports,
                        "calls": list(set(calls))
                    }
                
            except Exception as e:
                print(f"         ❌ Error tracing usage: {e}")
    
    def _audit_consciousness_components(self):
        """Verify completeness of consciousness components"""
        required_components = {
            "Memory": ["memory_timeline.py", "memory.py", "human_memory.py"],
            "Mood": ["mood_manager.py", "emotion.py"],
            "Goals": ["goal_manager.py", "goal_engine.py"],
            "Thoughts": ["thought_loop.py", "inner_monologue.py"],
            "Personality": ["personality_profile.py", "personality_state.py"],
            "Beliefs": ["belief_analyzer.py", "belief_evolution_tracker.py"]
        }
        
        print("   🧩 Checking consciousness component completeness...")
        
        component_status = {}
        
        for component, file_patterns in required_components.items():
            print(f"      🔍 {component}:")
            
            files_found = []
            for pattern in file_patterns:
                matches = list(self.ai_path.glob(f"**/{pattern}"))
                files_found.extend([str(m.relative_to(self.repo_path)) for m in matches])
            
            component_status[component] = {
                "files_found": files_found,
                "file_count": len(files_found),
                "is_available": len(files_found) > 0
            }
            
            if files_found:
                print(f"         ✅ Available ({len(files_found)} files)")
                for file_path in files_found:
                    print(f"            📄 {file_path}")
            else:
                print(f"         ❌ Missing")
                self.results["issues_found"].append(f"{component} consciousness component not found")
        
        # Calculate overall completeness
        available_components = sum(1 for comp in component_status.values() if comp["is_available"])
        total_components = len(required_components)
        completeness_percentage = (available_components / total_components) * 100
        
        component_status["overall_completeness"] = {
            "available": available_components,
            "total": total_components,
            "percentage": completeness_percentage,
            "is_complete": completeness_percentage >= 80
        }
        
        print(f"\n   📊 Overall Completeness: {available_components}/{total_components} ({completeness_percentage:.1f}%)")
        
        if completeness_percentage < 80:
            self.results["issues_found"].append(
                f"Consciousness components incomplete: {completeness_percentage:.1f}% ({available_components}/{total_components})"
            )
        
        self.results["consciousness_components"]["component_analysis"] = component_status
    
    def _generate_final_analysis(self):
        """Generate final analysis and recommendations"""
        
        # Calculate overall integration score
        scores = []
        
        # Class 5 verification score
        class5_files_with_integration = sum(
            1 for file_data in self.results["class5_verification"].values()
            if isinstance(file_data, dict) and file_data.get("uses_class5", False)
        )
        total_class5_files = len(self.results["class5_verification"])
        class5_score = (class5_files_with_integration / max(1, total_class5_files)) * 100
        scores.append(("Class 5 Integration", class5_score))
        
        # Prompt builder consciousness score
        if "prompt_builder_analysis" in self.results:
            consciousness_files = sum(
                1 for file_data in self.results["prompt_builder_analysis"].values()
                if isinstance(file_data, dict) and file_data.get("consciousness_integration", False)
            )
            total_prompt_files = len(self.results["prompt_builder_analysis"])
            prompt_score = (consciousness_files / max(1, total_prompt_files)) * 100
            scores.append(("Prompt Consciousness", prompt_score))
        
        # Component completeness score
        if "consciousness_components" in self.results and "component_analysis" in self.results["consciousness_components"]:
            completeness = self.results["consciousness_components"]["component_analysis"]["overall_completeness"]["percentage"]
            scores.append(("Component Completeness", completeness))
        
        # Overall integration score
        overall_score = sum(score for _, score in scores) / len(scores) if scores else 0
        
        # Determine integration status
        if overall_score >= 80:
            status = "EXCELLENT"
            status_emoji = "🟢"
        elif overall_score >= 60:
            status = "GOOD"
            status_emoji = "🟡"
        elif overall_score >= 40:
            status = "NEEDS_IMPROVEMENT"
            status_emoji = "🟠"
        else:
            status = "CRITICAL"
            status_emoji = "🔴"
        
        # Generate recommendations
        recommendations = []
        
        if class5_score < 80:
            recommendations.append("Implement Class 5 consciousness integration in all LLM handler functions")
        
        if any("get_consciousness_snapshot" in issue for issue in self.results["issues_found"]):
            recommendations.append("Implement or improve get_consciousness_snapshot() function to return complete consciousness state")
        
        if any("duplicate" in issue.lower() for issue in self.results["issues_found"]):
            recommendations.append("Consolidate duplicate prompt builders to avoid confusion and maintenance issues")
        
        if overall_score < 70:
            recommendations.append("Conduct comprehensive integration review to ensure all consciousness components are properly connected")
        
        self.results["integration_status"] = {
            "overall_score": overall_score,
            "status": status,
            "status_emoji": status_emoji,
            "component_scores": scores,
            "total_issues": len(self.results["issues_found"]),
            "recommendation_count": len(recommendations)
        }
        
        self.results["recommendations"] = recommendations
        
        # Print summary
        print(f"\n{status_emoji} Overall Integration Status: {status} ({overall_score:.1f}%)")
        print(f"   📊 Component Scores:")
        for component, score in scores:
            print(f"      • {component}: {score:.1f}%")
        print(f"   ⚠️ Issues Found: {len(self.results['issues_found'])}")
        print(f"   💡 Recommendations: {len(recommendations)}")
    
    def generate_detailed_non_consciousness_report(self) -> Dict[str, Any]:
        """Generate detailed report of files not using generate_response_with_consciousness"""
        
        non_consciousness_files = {}
        duplicate_builders = []
        
        # Analyze files that don't use Class 5 consciousness
        for file_path, analysis in self.results["class5_verification"].items():
            if isinstance(analysis, dict) and not analysis.get("uses_class5", False):
                if analysis.get("functions_found"):
                    non_consciousness_files[file_path] = {
                        "functions": analysis["functions_found"],
                        "alternative_prompt_builders": analysis.get("alternative_prompt_builders", []),
                        "direct_llm_calls": analysis.get("direct_llm_calls", []),
                        "instead_uses": self._determine_what_file_uses_instead(file_path, analysis)
                    }
        
        # Collect all duplicate prompt builders
        if "prompt_builder_analysis" in self.results and "builder_detection" in self.results["prompt_builder_analysis"]:
            builder_data = self.results["prompt_builder_analysis"]["builder_detection"]
            all_functions = builder_data.get("build_prompt_functions", [])
            
            # Group by function name to find duplicates
            function_groups = {}
            for func_name, file_path in all_functions:
                if func_name not in function_groups:
                    function_groups[func_name] = []
                function_groups[func_name].append(file_path)
            
            # Find actual duplicates and similar functions
            for func_name, files in function_groups.items():
                if len(files) > 1:
                    duplicate_builders.append({
                        "function_name": func_name,
                        "files": files,
                        "is_exact_duplicate": True
                    })
            
            # Add similar function names (build_*_prompt variations)
            similar_patterns = {}
            for func_name, files in function_groups.items():
                base_pattern = re.sub(r'_?(build|prompt)_?', '', func_name.lower())
                if 'prompt' in func_name.lower() and 'build' in func_name.lower():
                    if base_pattern not in similar_patterns:
                        similar_patterns[base_pattern] = []
                    similar_patterns[base_pattern].append((func_name, files[0]))
            
            for pattern, variations in similar_patterns.items():
                if len(variations) > 1:
                    duplicate_builders.append({
                        "pattern": pattern,
                        "variations": variations,
                        "is_exact_duplicate": False,
                        "type": "similar_functionality"
                    })
        
        return {
            "files_not_using_consciousness": non_consciousness_files,
            "duplicate_prompt_builders": duplicate_builders,
            "summary": {
                "total_non_consciousness_files": len(non_consciousness_files),
                "total_duplicate_patterns": len(duplicate_builders)
            }
        }
    
    def _determine_what_file_uses_instead(self, file_path: str, analysis: Dict[str, Any]) -> List[str]:
        """Determine what a file uses instead of consciousness integration"""
        alternatives = []
        
        # Check alternative prompt builders
        alt_builders = analysis.get("alternative_prompt_builders", [])
        if alt_builders:
            alternatives.extend([f"Alternative prompt builder: {builder}" for builder in set(alt_builders)])
        
        # Check direct LLM calls
        direct_calls = analysis.get("direct_llm_calls", [])
        if direct_calls:
            alternatives.extend([f"Direct LLM call: {call}" for call in set(direct_calls)])
        
        # Check for delegation patterns
        try:
            full_path = self.repo_path / file_path
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for imports and calls to other generate_response functions
            delegation_patterns = [
                r'from\s+ai\.chat\s+import\s+.*generate_response',
                r'generate_response_streaming\s*\(',
                r'from\s+ai\.llm_handler\s+import.*LLMHandler'
            ]
            
            for pattern in delegation_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    alternatives.extend([f"Delegates to: {match}" for match in matches])
                    
        except Exception:
            pass
        
        return alternatives if alternatives else ["Unknown - direct implementation"]


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Audit Class 5 Consciousness Integration")
    parser.add_argument("--repo-path", default=".", help="Path to repository root")
    parser.add_argument("--output", default="class5_audit_results.json", help="Output JSON file")
    parser.add_argument("--detailed-report", default="class5_detailed_report.json", help="Detailed report JSON file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize auditor
    auditor = Class5ConsciousnessAuditor(args.repo_path)
    
    # Run audit
    results = auditor.run_complete_audit()
    
    # Generate detailed report for non-consciousness usage
    detailed_report = auditor.generate_detailed_non_consciousness_report()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(args.detailed_report, 'w') as f:
        json.dump(detailed_report, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output}")
    print(f"💾 Detailed report saved to: {args.detailed_report}")
    
    # Print detailed summary for user
    print("\n" + "="*80)
    print("📋 DETAILED REPORT: Files NOT using generate_response_with_consciousness()")
    print("="*80)
    
    for file_path, details in detailed_report["files_not_using_consciousness"].items():
        print(f"\n📁 {file_path}")
        print(f"   Functions: {', '.join(details['functions'])}")
        print(f"   Uses instead:")
        for alternative in details['instead_uses']:
            print(f"      • {alternative}")
    
    print("\n" + "="*80)
    print("🔄 DUPLICATE PROMPT BUILDERS")
    print("="*80)
    
    for duplicate in detailed_report["duplicate_prompt_builders"]:
        if duplicate.get("is_exact_duplicate", False):
            print(f"\n❌ Exact duplicate: {duplicate['function_name']}")
            for file_path in duplicate['files']:
                print(f"      📁 {file_path}")
        else:
            print(f"\n⚠️ Similar functionality: {duplicate.get('pattern', 'unknown')}")
            for func_name, file_path in duplicate.get('variations', []):
                print(f"      📁 {file_path}: {func_name}")
    
    print(f"\n📊 Summary:")
    print(f"   • Files not using consciousness: {detailed_report['summary']['total_non_consciousness_files']}")
    print(f"   • Duplicate prompt patterns: {detailed_report['summary']['total_duplicate_patterns']}")
    
    # Return exit code based on critical issues
    critical_issues = len(results["issues_found"])
    if critical_issues > 0:
        print(f"⚠️ {critical_issues} issues found - see report for details")
        return 1
    else:
        print("✅ No critical issues found")
        return 0


if __name__ == "__main__":
    sys.exit(main())