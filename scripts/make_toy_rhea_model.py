"""Generate a small toy TorchScript model for the Rhea flavor-mixing test inputs.

Exports a predict_all(f4_in) -> (F4_out, growthrate, stability) method matching the
contract RheaModel::Predict expects (src/radiation_m1/radiation_m1_rhea.hpp/.cpp):
f4_in is [n, 2, 3, 4] float32; F4_out is [n, 2, 3, 4] float32; growthrate and
stability are [n] float32.

Both models below are deliberately row-independent (no batch-mixing op such as
batchnorm or a softmax over dim 0): every output row depends only on its own input
row, mirroring the per-zone independence of the real model. This matters for testing
runs whose active batch extent varies (e.g. across AMR regrids): per-row results at a
smaller batch size must be comparable against a full-batch run, which only holds if
rows never mix. Both also return contiguous tensors, which RheaModel::Predict assumes
when it wraps the outputs in unmanaged Kokkos Views.

Two model variants, selected with --model:

  generic
      Input-dependent but physically inert: a small nonlinearity plus a per-row scalar
      reduction, so a broken pack kernel or a wrong batch slice is visible in the
      output rather than masked by a constant response. Note it reports every cell
      STABLE (stability = 1.0 for any positive-density input), which gates the BGK
      rate to zero and makes ApplyRheaMixing an exact bit-for-bit no-op. That is the
      right stand-in for a plumbing/AMR/regression smoke test -- and it is what makes
      those tests bit-reproducible -- but it means such a run exercises pack ->
      predict -> unpack and nothing downstream of the stability gate.

  swap (default)
      Flags every cell UNSTABLE and performs a full e<->x (and ebar<->xbar) flavor
      swap, so ReconstructMixingMatrix recovers a survival probability p = 0 and the
      BGK relaxation actually runs. Use this to test the mixing path itself. The
      growth rate is a constant chosen so the resulting BGK rate is exactly
      --gamma-code per unit code time, which makes the update analytically
      predictable: with a full swap, lambda = exp(-dt*alpha*gamma_code*tau_factor)
      and N_new = lambda*N_old + (1 - lambda)*N_swapped. Pick --gamma-code so that
      dt*alpha*gamma_code is O(1); a much larger value saturates lambda to 0 and
      collapses the test to an instantaneous swap, which is insensitive to the rate.

      This is the default: every in-tree input now wants swap, and a forgotten
      --model flag should produce a sensitive test (mixing actually runs, so a
      regression is visible) rather than a silently vacuous one (generic's
      always-stable no-op, which would pass even a broken mixing kernel).

These are test stand-ins with no physical content. Usage:

    python3 make_toy_rhea_model.py [output_path.pt] [--model generic|swap]
                                   [--gamma-code FLOAT]

Generate the file named by rhea_model_path in the chosen input file and place it in
the working directory the run starts from. As of this writing: both TOV inputs
(inputs/tests/rad_m1_tov_rhea.athinput and rad_m1_tov_rhea_amr.athinput) want
toy_flavor_swap_tov.pt built with --gamma-code 1.0; the single-zone test
(inputs/tests/rad_m1_rhea_singlezone.athinput) wants toy_flavor_swap.pt built at the
default --gamma-code 10.0. Both now use the default --model swap.
"""
import argparse
import torch

# Conversion from Rhea's growthrate output (a number density, in code units) to a rate
# in 1/code_time. MUST stay in sync with growthrate_to_code in ApplyRheaMixing
# (src/radiation_m1/radiation_m1_flavor_mix_rhea.cpp), which builds the same product as
#   MakeGeometricSolar().NumberDensityConversion(MakeCGS())  # 1e39, cm^-3 per fm^-3
#   * 1.9255158167467008e-22                                 # sqrt(2)*G_F/hbar, cm^3/s
#   * MakeGeometricSolar().TimeConversion(MakeCGS())         # G*Msun/c^3, s per code time
# If that factor ever changes, the --gamma-code calibration below silently stops
# holding and the swap model's analytic prediction goes with it.
GROWTHRATE_TO_CODE = 9.484132591648745e11


class ToyRhea(torch.nn.Module):
    """Physically inert, input-dependent; reports every cell stable."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unused entry point -- the code freezes and calls only "predict_all"
        # (torch::jit::freeze_module(model, {"predict_all"})), never forward.
        return x

    @torch.jit.export
    def predict_all(self, f4_in: torch.Tensor):
        f4_out = f4_in + 0.01 * torch.tanh(f4_in)
        flat = f4_in.reshape(f4_in.shape[0], -1)
        growthrate = flat.sum(dim=1)
        stability = (flat.sum(dim=1) > 0).to(torch.float32)
        return f4_out, growthrate, stability


class ToyRheaSwap(torch.nn.Module):
    """Unstable everywhere, full flavor swap, constant calibrated growth rate."""

    growthrate_raw: float

    def __init__(self, growthrate_raw: float):
        super().__init__()
        self.growthrate_raw = growthrate_raw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @torch.jit.export
    def predict_all(self, f4_in: torch.Tensor):
        # Slot f=0 is nu_e (or nu_ebar); slots f=1,2 are the heavy channel, which the
        # pack kernel split into equal halves. Swapping means the outgoing e slot gets
        # the summed heavy density and each heavy slot gets half the incoming e slot --
        # the exact inverse of the i_flv_map/flv_fac fold-in, so the fold-out in
        # ApplyRheaMixing recovers a clean p = 0.
        e = f4_in[:, :, 0:1, :]
        x = f4_in[:, :, 1:3, :].sum(dim=2, keepdim=True)
        f4_out = torch.cat([x, (e / 2.0).expand(-1, -1, 2, -1)], dim=2).contiguous()
        n = f4_in.shape[0]
        growthrate = torch.full((n,), self.growthrate_raw,
                                dtype=f4_in.dtype, device=f4_in.device)
        # 0.0 == unstable, so the BGK rate is taken from growthrate above rather than
        # being gated to zero.
        stability = torch.zeros((n,), dtype=f4_in.dtype, device=f4_in.device)
        return f4_out, growthrate, stability


def main():
    parser = argparse.ArgumentParser(
        description="Generate a toy TorchScript stand-in for the Rhea model.")
    parser.add_argument("output_path", nargs="?", default="toy_flavor_swap.pt",
                        help="output .pt path (default: toy_flavor_swap.pt)")
    parser.add_argument("--model", choices=("generic", "swap"), default="swap",
                        help="model variant (default: swap)")
    parser.add_argument("--gamma-code", type=float, default=10.0,
                        help="swap model only: target BGK rate in 1/code_time "
                             "(default: 10.0)")
    args = parser.parse_args()

    if args.model == "swap":
        growthrate_raw = args.gamma_code / GROWTHRATE_TO_CODE
        model = ToyRheaSwap(growthrate_raw)
        note = (" (growthrate_raw={:.9e} -> gamma_code={:g} per code time)"
                .format(growthrate_raw, args.gamma_code))
    else:
        model = ToyRhea()
        note = " (stability=1.0 everywhere: mixing is an exact no-op)"

    torch.jit.script(model).save(args.output_path)
    print("saved", args.output_path, "[--model {}]".format(args.model) + note)


if __name__ == "__main__":
    main()
