# Directory Guide

## Role
Shearing-box boundary exchange, orbital advection, and associated source terms.

## Important Files
- `shearing_box.hpp`: abstract and concrete shearing-box boundary classes for cell- and face-centered data.
- `shearing_box.cpp`, `shearing_box_cc.cpp`, `shearing_box_fc.cpp`: implementation of boundary communication and unpacking.
- `shearing_box_srcterms.cpp`, `shearing_box_tasks.cpp`: source-term application and task integration.
- `orbital_advection.hpp/.cpp`, `orbital_advection_cc.cpp`, `orbital_advection_fc.cpp`, `orbital_advection_tasks.cpp`: orbital-advection-specific paths; both the CC and FC remap paths now support an explicit remap timestep for the sweep-final STS remap.
- `remap_fluxes.hpp`: shared remap helpers.

## Read This Next
- For corresponding runtime examples, inspect `inputs/shearing_box/AGENTS.md`.
- Hydro STS is now compatible with the shearing-box/orbital-advection path: the Hydro
  parabolic loop runs orbital remap once per sweep with `dt_sweep`, while x1 shearing-box
  exchange runs every STS stage.
- MHD STS is now compatible with the shearing-box/orbital-advection path as well: both
  CC and FC orbital remaps run once per sweep with `dt_sweep`, while U/B x1 shearing-box
  exchange runs every STS stage.
- `ion-neutral` STS is still fenced; read `src/driver/AGENTS.md` and
  `doc/sts_implementation_guide.md` before trying to widen that scope.
