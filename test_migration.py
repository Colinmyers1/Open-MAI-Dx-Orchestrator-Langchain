#!/usr/bin/env python3
"""
Simple test script to verify the LangGraph migration worked correctly.

This script does basic checks without requiring the dependencies to be installed.
"""

import sys
import os
import ast
import importlib.util

def check_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        ast.parse(source)
        print(f"✅ {file_path}: Syntax OK")
        return True
    except SyntaxError as e:
        print(f"❌ {file_path}: Syntax Error - {e}")
        return False
    except Exception as e:
        print(f"❌ {file_path}: Error - {e}")
        return False

def check_file_exists(file_path):
    """Check if a file exists."""
    if os.path.exists(file_path):
        print(f"✅ {file_path}: File exists")
        return True
    else:
        print(f"❌ {file_path}: File missing")
        return False

def check_imports_in_file(file_path, expected_imports):
    """Check if a file contains expected imports."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        missing_imports = []
        for imp in expected_imports:
            if imp not in content:
                missing_imports.append(imp)
        
        if not missing_imports:
            print(f"✅ {file_path}: All expected imports found")
            return True
        else:
            print(f"❌ {file_path}: Missing imports - {missing_imports}")
            return False
    except Exception as e:
        print(f"❌ {file_path}: Error checking imports - {e}")
        return False

def main():
    """Run all migration checks."""
    print("🔍 Testing LangGraph Migration...")
    print("=" * 50)
    
    all_checks_passed = True
    
    # 1. Check that key files exist
    print("\n📁 File Existence Checks:")
    files_to_check = [
        "mai_dx/state.py",
        "mai_dx/tools.py", 
        "mai_dx/graph.py",
        "mai_dx/main.py",
        "mai_dx/agents/__init__.py",
        "mai_dx/agents/hypothesis.py",
        "mai_dx/agents/consensus.py",
        "mai_dx/agents/gatekeeper.py",
        "requirements.txt",
        "pyproject.toml",
    ]
    
    for file_path in files_to_check:
        if not check_file_exists(file_path):
            all_checks_passed = False
    
    # 2. Check syntax of Python files
    print("\n🐍 Syntax Checks:")
    python_files = [
        "mai_dx/state.py",
        "mai_dx/tools.py",
        "mai_dx/graph.py", 
        "mai_dx/main.py",
        "mai_dx/agents/hypothesis.py",
        "mai_dx/agents/consensus.py",
        "mai_dx/agents/gatekeeper.py",
        "example.py",
    ]
    
    for file_path in python_files:
        if os.path.exists(file_path):
            if not check_syntax(file_path):
                all_checks_passed = False
    
    # 3. Check key imports
    print("\n📦 Import Checks:")
    
    # Check that main.py uses LangGraph (indirectly through modules)
    langgraph_imports = [
        "DiagnosticState",
        "compile_diagnostic_graph",
    ]
    if not check_imports_in_file("mai_dx/main.py", langgraph_imports):
        all_checks_passed = False
    
    # Check that Swarms is removed from dependencies
    print("\n🗑️  Dependency Migration Checks:")
    
    # Check requirements.txt
    try:
        with open("requirements.txt", 'r') as f:
            reqs = f.read()
        
        if "swarms" in reqs:
            print("❌ requirements.txt: Still contains 'swarms'")
            all_checks_passed = False
        else:
            print("✅ requirements.txt: 'swarms' removed")
            
        if "langgraph" in reqs:
            print("✅ requirements.txt: 'langgraph' added")
        else:
            print("❌ requirements.txt: Missing 'langgraph'")
            all_checks_passed = False
            
    except Exception as e:
        print(f"❌ requirements.txt: Error checking - {e}")
        all_checks_passed = False
    
    # 4. Check that backup was created
    print("\n💾 Backup Checks:")
    if check_file_exists("mai_dx/main_swarms_backup.py"):
        print("✅ Swarms backup created successfully")
    else:
        print("⚠️  No Swarms backup found (mai_dx/main_swarms_backup.py)")
    
    # 5. Architecture checks
    print("\n🏗️  Architecture Checks:")
    
    # Check that state management is properly structured
    if os.path.exists("mai_dx/state.py"):
        with open("mai_dx/state.py", 'r') as f:
            state_content = f.read()
        
        state_checks = [
            ("DiagnosticState", "DiagnosticState defined"),
            ("TypedDict", "Using TypedDict for state"),
            ("add_messages", "Using LangGraph message management"),
        ]
        
        for check, description in state_checks:
            if check in state_content:
                print(f"✅ State: {description}")
            else:
                print(f"❌ State: Missing {description}")
                all_checks_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 All migration checks passed!")
        print("✅ LangGraph migration appears successful")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up environment variables (API keys)")
        print("3. Run: python example.py")
    else:
        print("❌ Some migration checks failed")
        print("⚠️  Please review the issues above")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())