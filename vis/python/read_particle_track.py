#!/usr/bin/env python3
"""Read AthenaK particle track files."""

from __future__ import annotations

import argparse
import struct
import warnings
from pathlib import Path

import numpy as np


FILE_HEADER = struct.Struct("=32s10i")
BLOCK_PREFIX = struct.Struct("=16s5i")
BLOCK_MAGIC = b"ATHKPARTBLOCK".ljust(16, b"\0")
BLOCK_END = b"ATHKPARTEND".ljust(16, b"\0")
SCAN_SIZE = 1024 * 1024


def _read_exact(handle, size: int, description: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"truncated {description}")
    return data


def _read_text(handle, size: int, description: str) -> str:
    if size < 0:
        raise ValueError(f"invalid {description} size")
    return _read_exact(handle, size, description).decode()


def _complete_block_at(handle, offset, file_size, version, real_size,
                       n_int_fields, n_real_fields):
    """Return block metadata when a complete block begins at offset."""
    if offset + BLOCK_PREFIX.size + real_size + len(BLOCK_END) > file_size:
        return None
    handle.seek(offset)
    prefix = handle.read(BLOCK_PREFIX.size)
    if len(prefix) != BLOCK_PREFIX.size:
        return None
    block_magic, block_version, nrecords, int_per, real_per, cycle = \
        BLOCK_PREFIX.unpack(prefix)
    if block_magic != BLOCK_MAGIC or block_version != version or nrecords < 0 or \
            int_per != n_int_fields or real_per != n_real_fields:
        return None

    time_offset = offset + BLOCK_PREFIX.size
    payload_offset = time_offset + real_size
    footer_offset = payload_offset + nrecords * (
        n_int_fields * 4 + n_real_fields * real_size
    )
    block_end = footer_offset + len(BLOCK_END)
    if block_end > file_size:
        return None
    handle.seek(footer_offset)
    if handle.read(len(BLOCK_END)) != BLOCK_END:
        return None
    return nrecords, cycle, time_offset, payload_offset, block_end


def _find_next_complete_block(handle, offset, file_size, version, real_size,
                              n_int_fields, n_real_fields):
    """Find the next complete block without loading the particle payload."""
    overlap = len(BLOCK_MAGIC) - 1
    carry = b""
    position = offset
    while position < file_size:
        handle.seek(position)
        chunk = handle.read(min(SCAN_SIZE, file_size-position))
        if not chunk:
            return None
        data = carry + chunk
        data_offset = position - len(carry)
        search_from = 0
        while True:
            found = data.find(BLOCK_MAGIC, search_from)
            if found < 0:
                break
            candidate = data_offset + found
            if candidate >= offset:
                block = _complete_block_at(
                    handle, candidate, file_size, version, real_size,
                    n_int_fields, n_real_fields
                )
                if block is not None:
                    return candidate, block
            search_from = found + 1
        carry = data[-overlap:]
        position += len(chunk)
    return None


def read_particle_track(path: str | Path) -> dict[str, np.ndarray]:
    """Return all particle records as named NumPy columns."""
    path = Path(path)
    columns: dict[str, list[np.ndarray]] = {"time": [], "cycle": []}

    with path.open("rb") as handle:
        raw_header = _read_exact(handle, FILE_HEADER.size, "file header")
        magic, version, real_size, int_size, population_size, type_size, tags_size, \
            int_names_size, real_names_size, n_int_fields, n_real_fields = \
            FILE_HEADER.unpack(raw_header)
        if magic.rstrip(b"\0") != b"ATHENAK_PARTICLE_TRACK":
            raise ValueError(f"{path} is not an AthenaK particle track file")
        if version != 1:
            raise ValueError(f"unsupported particle track version {version}")
        if int_size != 4:
            raise ValueError(f"unsupported integer size {int_size}")
        if real_size not in (4, 8):
            raise ValueError(f"unsupported Real size {real_size}")
        if n_int_fields < 1 or n_real_fields < 0:
            raise ValueError("invalid field counts in file header")

        # These strings identify and validate an appended file.  Field names define the
        # columns returned below; the other values do not need special reader behavior.
        _read_text(handle, population_size, "population name")
        _read_text(handle, type_size, "particle type")
        _read_text(handle, tags_size, "tag selection")
        int_names_blob = _read_text(handle, int_names_size, "integer field list")
        real_names_blob = _read_text(handle, real_names_size, "real field list")
        int_names = int_names_blob.split("\n") if int_names_blob else []
        real_names = real_names_blob.split("\n") if real_names_blob else []
        if len(int_names) != n_int_fields or len(real_names) != n_real_fields:
            raise ValueError("field list length does not match file header")
        field_names = int_names + real_names
        if len(set(field_names + ["time", "cycle"])) != len(field_names) + 2:
            raise ValueError("particle track field names are not unique")
        for name in field_names:
            columns[name] = []

        int_dtype = np.dtype("=i4")
        real_dtype = np.dtype("=f8" if real_size == 8 else "=f4")
        data_start = handle.tell()
        handle.seek(0, 2)
        file_size = handle.tell()
        block_offset = data_start
        while block_offset < file_size:
            block = _complete_block_at(
                handle, block_offset, file_size, version, real_size,
                n_int_fields, n_real_fields
            )
            if block is None:
                recovered = _find_next_complete_block(
                    handle, block_offset+1, file_size, version, real_size,
                    n_int_fields, n_real_fields
                )
                if recovered is None:
                    warnings.warn(
                        f"{path}: ignoring incomplete particle track data beginning "
                        f"at byte {block_offset}", RuntimeWarning
                    )
                    break
                recovered_offset, block = recovered
                warnings.warn(
                    f"{path}: skipping malformed particle track data from byte "
                    f"{block_offset} to byte {recovered_offset}", RuntimeWarning
                )
                block_offset = recovered_offset

            nrecords, cycle, time_offset, payload_offset, block_end = block
            handle.seek(time_offset)
            time = np.frombuffer(
                _read_exact(handle, real_size, "particle track block time"),
                dtype=real_dtype,
                count=1,
            )[0]
            handle.seek(payload_offset)
            ints = np.fromfile(
                handle, dtype=int_dtype, count=nrecords * n_int_fields
            )
            reals = np.fromfile(
                handle, dtype=real_dtype, count=nrecords * n_real_fields
            )
            if ints.size != nrecords * n_int_fields or \
                    reals.size != nrecords * n_real_fields:
                raise ValueError("truncated particle track block payload")
            ints = ints.reshape(nrecords, n_int_fields)
            reals = reals.reshape(nrecords, n_real_fields)
            columns["time"].append(np.full(nrecords, time, dtype=real_dtype))
            columns["cycle"].append(np.full(nrecords, cycle, dtype=int_dtype))
            for n, name in enumerate(int_names):
                columns[name].append(ints[:, n].copy())
            for n, name in enumerate(real_names):
                columns[name].append(reals[:, n].copy())
            block_offset = block_end

    result = {}
    for name, arrays in columns.items():
        if arrays:
            result[name] = np.concatenate(arrays)
        elif name in {"cycle", *int_names}:
            result[name] = np.empty(0, dtype=np.int32)
        else:
            result[name] = np.empty(0, dtype=real_dtype)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to a *.part_track file")
    parser.add_argument("--npz", help="Optional output .npz file")
    args = parser.parse_args()

    data = read_particle_track(args.path)
    print(f"records: {len(data['time'])}")
    print("columns:", " ".join(data))
    if args.npz:
        np.savez(args.npz, **data)


if __name__ == "__main__":
    main()
