"""Compatibility entry point for 07_llm-evaluate.py."""
import runpy
from pathlib import Path
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("07_llm-evaluate.py")), run_name="__main__")
