"""
Executable entrypoint when running from parent directory or repo root.
"""
import sys
from pathlib import Path

proj_dir = Path(__file__).resolve().parent

# If Python initially registered outer folder as a namespace package, purge it
# so the true cctv_pipeline package inside proj_dir is imported cleanly.
if "cctv_pipeline" in sys.modules:
    del sys.modules["cctv_pipeline"]

if str(proj_dir) not in sys.path:
    sys.path.insert(0, str(proj_dir))

from cctv_pipeline.cli import main

if __name__ == "__main__":
    main()
