# Directory Guide

## Role
All runtime output configuration and format writers. This directory is the choke point for adding a new file type or exported variable.

## Important Files
- `outputs.hpp`, `outputs.cpp`: core output object, `<output[n]>` parsing, variable registry, and writer dispatch.
- `io_wrapper.hpp/.cpp`: abstraction over file I/O used by outputs and restarts.
- `derived_variables.cpp`: computed quantities available to output blocks.
- `history.cpp`, `eventlog.cpp`, `formatted_table.cpp`: text-based diagnostics and histories.
- `binary.cpp`, `coarsened_binary.cpp`, `cartgrid.cpp`, `vtk_mesh.cpp`, `vtk_prtcl.cpp`, `spherical_surface.cpp`: major data exporters.
- `restart.cpp`: restart serialization/deserialization.
- `track_prtcl.cpp`, `pdf.cpp`: particle tracking and PDF outputs.

## Read This Next
- For runtime configuration of outputs, also inspect example `inputs/*.athinput` files.
