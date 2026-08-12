#!/usr/bin/env python3
# AthenaK astrophysical plasma code
# Copyright(C) 2020 James M. Stone and the AthenaK collaboration
# Licensed under the 3-clause BSD License (the "LICENSE")

"""Compare Celephais A/B output payloads while normalising toggle metadata."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


IGNORED_NAMES = {
    "output.sha256",
    "reference.sha256",
    "run.log",
    "semantic_output.sha256",
}
NORMALISED_PARAMETER_KEYS = {b"batch_fields", b"skip_vacuum_velocity"}
PARAMETER_HEADER_SUFFIXES = {".bin", ".rst"}
# restart.cpp writes this native fixed header immediately after <par_end>:
# 3 ints, 2 Reals, RegionSize, and 2 RegionIndcs.  This BNS harness builds
# double-precision AthenaK, so it is 252 bytes.  Mesh construction leaves the
# root mesh's unused coarse-index members indeterminate; normalise only those
# 36 bytes while retaining every meaningful fixed-header and state byte.
RESTART_FIXED_HEADER_BYTES = 252
MESH_INDCS_COARSE_BEGIN = 120
MESH_INDCS_COARSE_END = 156


def update_file_digest(path: Path, digest: "hashlib._Hash") -> int:
    total_bytes = path.stat().st_size
    with path.open("rb") as stream:
        if path.suffix in PARAMETER_HEADER_SUFFIXES:
            header_bytes = 0
            while True:
                line = stream.readline()
                if not line:
                    raise RuntimeError(f"{path}: missing <par_end> in parameter header")
                header_bytes += len(line)
                if header_bytes > 1024 * 1024:
                    raise RuntimeError(f"{path}: parameter header exceeds 1 MiB")
                original_line = line
                key, separator, _ = line.partition(b"=")
                if separator and key.strip() in NORMALISED_PARAMETER_KEYS:
                    line = key.strip() + b" = <normalised A/B toggle>\n"
                digest.update(line)
                if original_line.strip() == b"<par_end>":
                    break

            if path.suffix == ".rst":
                fixed_header = bytearray(stream.read(RESTART_FIXED_HEADER_BYTES))
                if len(fixed_header) != RESTART_FIXED_HEADER_BYTES:
                    raise RuntimeError(f"{path}: truncated fixed restart header")
                fixed_header[MESH_INDCS_COARSE_BEGIN:MESH_INDCS_COARSE_END] = bytes(
                    MESH_INDCS_COARSE_END - MESH_INDCS_COARSE_BEGIN
                )
                digest.update(fixed_header)

        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return total_bytes


def manifest(root: Path) -> tuple[dict[Path, str], int]:
    if not root.is_dir():
        raise RuntimeError(f"not a run directory: {root}")
    result: dict[Path, str] = {}
    total_bytes = 0
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.name in IGNORED_NAMES or path.name.startswith("gmon."):
            continue
        relative = path.relative_to(root)
        digest = hashlib.sha256()
        total_bytes += update_file_digest(path, digest)
        result[relative] = digest.hexdigest()
    if not result:
        raise RuntimeError(f"no output files found under {root}")
    return result, total_bytes


def main(arguments: list[str]) -> int:
    if len(arguments) < 3:
        print(f"usage: {arguments[0]} REFERENCE RUN [RUN ...]", file=sys.stderr)
        return 2

    reference_root = Path(arguments[1]).resolve()
    reference, reference_bytes = manifest(reference_root)
    print(
        f"[celephais-output] reference={reference_root} files={len(reference)} "
        f"bytes={reference_bytes}"
    )

    for argument in arguments[2:]:
        root = Path(argument).resolve()
        observed, observed_bytes = manifest(root)
        if observed.keys() != reference.keys():
            missing = sorted(str(path) for path in reference.keys() - observed.keys())
            extra = sorted(str(path) for path in observed.keys() - reference.keys())
            raise RuntimeError(f"{root}: file-set mismatch; missing={missing}, extra={extra}")
        mismatches = [path for path in reference if reference[path] != observed[path]]
        if mismatches:
            raise RuntimeError(f"{root}: payload mismatch in {mismatches}")
        print(
            f"[celephais-output] match={root} files={len(observed)} "
            f"bytes={observed_bytes}"
        )
    print("[celephais-output] all normalised production payloads are identical")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
