"""Compile native C2P failure regressions with an existing Unix Makefiles build.

Usage: python tst/test_c2p_failure.py /absolute/path/to/build
Requires CMAKE_EXPORT_COMPILE_COMMANDS=ON and built Kokkos libraries. The
selected compiler/backend is reused; a Serial run is not a GPU verification.
"""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


def main():
    build = Path(sys.argv[1]).resolve()
    root = Path(__file__).resolve().parent.parent
    source = root / "tst/test_suite/unit_tests/c2p_failure.cpp"
    units = root / "src/eos/primitive-solver/unit_system.cpp"
    entries = json.loads((build / "compile_commands.json").read_text())
    entry = next(e for e in entries if e["file"].endswith("/src/main.cpp"))
    flags = shlex.split(entry["command"])
    # Reuse backend flags, replacing only the original input/output selection.
    compile_flags = flags[:flags.index("-o")]
    link = shlex.split((build / "src/CMakeFiles/athena.dir/link.txt").read_text())
    libraries = link[link.index("athena") + 1:]
    with tempfile.TemporaryDirectory(prefix="hermes-verify-c2p-") as tmp:
        exe = Path(tmp) / "c2p_failure"
        subprocess.run(compile_flags + [str(source), str(units), "-o", str(exe)]
                       + libraries, cwd=entry["directory"], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
