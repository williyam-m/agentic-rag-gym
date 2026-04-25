"""Deploy Agentic RAG OS to HuggingFace Spaces."""
import os
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent

def main():
    api = HfApi()
    repo_id = "williyam/agentic-rag-os"

    # Create temp deploy dir
    deploy_dir = Path(tempfile.mkdtemp())
    print(f"Deploy dir: {deploy_dir}")

    # Copy source files
    for folder in ["rag_master", "server", "domains", "agentic_rag_os"]:
        src = ROOT / folder
        dst = deploy_dir / folder
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.db", "data"))
        print(f"  Copied {folder}/")

    # Copy pyproject.toml
    shutil.copy(ROOT / "pyproject.toml", deploy_dir / "pyproject.toml")

    # Use agentic_rag_os Dockerfile
    shutil.copy(ROOT / "agentic_rag_os" / "Dockerfile", deploy_dir / "Dockerfile")

    # Use HF README
    shutil.copy(ROOT / "agentic_rag_os" / "HF_README.md", deploy_dir / "README.md")

    print(f"\nFiles in deploy dir:")
    for f in sorted(deploy_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(deploy_dir)
            print(f"  {rel}")

    # Upload to HF Space
    print(f"\nUploading to {repo_id}...")
    api.upload_folder(
        folder_path=str(deploy_dir),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy Agentic RAG OS v1.0.0",
    )
    print(f"\nDone! Space: https://huggingface.co/spaces/{repo_id}")

    # Cleanup
    shutil.rmtree(deploy_dir)

if __name__ == "__main__":
    main()
