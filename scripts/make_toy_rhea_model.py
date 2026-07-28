"""Generate a small toy TorchScript model for the Rhea flavor-mixing test inputs.

Exports a predict_all(f4_in) -> (F4_out, growthrate, stability) method matching the
contract RheaModel::Predict expects (src/radiation_m1/radiation_m1_rhea.hpp/.cpp):
f4_in is [n, 2, 3, 4] float32; F4_out is [n, 2, 3, 4] float32; growthrate and
stability are [n] float32. The model is genuinely input-dependent (a small
nonlinearity plus a per-row scalar reduction), so a broken pack kernel or a wrong
batch slice is visible in the output rather than masked by a constant response.

Deliberately row-independent (no batch-mixing op such as batchnorm or a softmax over
dim 0): every output row depends only on its own input row, mirroring the per-zone
independence of the real model. This matters for testing runs whose active batch
extent varies (e.g. across AMR regrids): per-row results at a smaller batch size must
be comparable against a full-batch run, which only holds if rows never mix.

This is a plumbing/smoke-test stand-in with no physical content. Usage:

    python3 make_toy_rhea_model.py [output_path.pt]

Generate the file named by rhea_model_path in the chosen input file (e.g.
toy_flavor_swap_tov.pt for inputs/tests/rad_m1_tov_rhea_amr.athinput) and place it in
the working directory the run starts from.
"""
import sys
import torch


class ToyRhea(torch.nn.Module):
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


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "toy_flavor_swap.pt"
    m = torch.jit.script(ToyRhea())
    m.save(out_path)
    print("saved", out_path)
