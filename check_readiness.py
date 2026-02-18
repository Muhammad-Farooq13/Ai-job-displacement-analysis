#!/usr/bin/env python
"""
GitHub Upload Readiness Checker

Verifies all necessary components are in place before uploading to GitHub
"""

import os
from pathlib import Path
import json

def check_readiness():
    """Check if project is ready for GitHub upload."""
    
    project_root = Path(__file__).parent
    
    print("\n" + "="*80)
    print("GITHUB UPLOAD READINESS CHECK")
    print("="*80)
    
    checks = {
        "✅ Code Structure": {
            "src/ directory": (project_root / "src").exists(),
            "tests/ directory": (project_root / "tests").exists(),
            "notebooks/ directory": (project_root / "notebooks").exists(),
            "deployment/ directory": (project_root / "deployment").exists(),
        },
        "✅ Configuration Files": {
            "setup.py": (project_root / "setup.py").exists(),
            "requirements.txt": (project_root / "requirements.txt").exists(),
            ".gitignore": (project_root / ".gitignore").exists(),
            "LICENSE": (project_root / "LICENSE").exists(),
        },
        "✅ Documentation": {
            "README.md": (project_root / "README.md").exists(),
            "CONTRIBUTING.md": (project_root / "CONTRIBUTING.md").exists(),
        },
        "✅ Deployment": {
            "Dockerfile": (project_root / "deployment" / "Dockerfile").exists(),
            "docker-compose.yml": (project_root / "deployment" / "docker-compose.yml").exists(),
            "FastAPI app.py": (project_root / "deployment" / "app.py").exists(),
        },
        "✅ CI/CD": {
            ".github/workflows/ci.yml": (project_root / ".github" / "workflows" / "ci.yml").exists(),
            ".github/workflows/tests.yml": (project_root / ".github" / "workflows" / "tests.yml").exists(),
        },
        "✅ Trained Models": {
            "Model artifact": (project_root / "models" / "trained_models" / "ai_replacement_predictor.pkl").exists(),
            "Model metadata": (project_root / "models" / "model_metadata.json").exists(),
            "Feature scaler": (project_root / "models" / "model_artifacts" / "feature_scaler.pkl").exists(),
        },
        "✅ Author Information": {
            "setup.py has author": check_file_contains(project_root / "setup.py", "Muhammad Farooq"),
            "setup.py has email": check_file_contains(project_root / "setup.py", "mfarooqshafee333@gmail.com"),
            "setup.py has GitHub URL": check_file_contains(project_root / "setup.py", "Muhammad-Farooq-13"),
            "README has author": check_file_contains(project_root / "README.md", "Muhammad Farooq"),
        }
    }
    
    all_passed = True
    for category, items in checks.items():
        print(f"\n{category}")
        for item, status in items.items():
            symbol = "✓" if status else "✗"
            print(f"  [{symbol}] {item}")
            if not status:
                all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ PROJECT IS READY FOR GITHUB UPLOAD!")
        print("="*80)
        print("\nNext Steps:")
        print("  1. Initialize Git repository:")
        print("     git init")
        print("\n  2. Add all files:")
        print("     git add .")
        print("\n  3. Create initial commit:")
        print("     git commit -m 'Initial commit: Production-ready ML project'")
        print("\n  4. Create GitHub repository at https://github.com/new")
        print("     Repository name: ai-job-displacement-analysis")
        print("\n  5. Add remote and push:")
        print("     git remote add origin https://github.com/Muhammad-Farooq-13/ai-job-displacement-analysis.git")
        print("     git branch -M main")
        print("     git push -u origin main")
        print("\n✨ Your project is ready to impress employers!")
    else:
        print("❌ SOME ITEMS ARE MISSING - CHECK ABOVE")
        print("="*80)
    
    return all_passed


def check_file_contains(filepath, text):
    """Check if a file contains specific text."""
    if not filepath.exists():
        return False
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            return text in content
    except:
        return False


if __name__ == "__main__":
    success = check_readiness()
    exit(0 if success else 1)
