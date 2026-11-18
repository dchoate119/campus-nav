"""Export the trained PyTorch step counter (Gait Wearable/models/step_counter.pt) to plain
NumPy arrays, so the UNO Q can run it with no torch (and no laptop, no wifi).

BatchNorm is folded into the convolutions, so the board only does conv + ReLU + add.
Run from anywhere:  python export_stepnet.py
Writes ../python/stepnet.npz and a parity check against torch.
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
GAIT = HERE.parent / "Gait Wearable"
if not GAIT.exists():                       # standalone copy of this project: use the bundled source
    GAIT = HERE / "stepcounter_source"
sys.path.insert(0, str(GAIT))
import StepCounter as sc  # noqa: E402

OUT = HERE.parent / "python" / "stepnet.npz"


def fold(conv, bn):
    w = conv.weight.detach().numpy().astype(np.float64)
    b = conv.bias.detach().numpy().astype(np.float64)
    s = bn.weight.detach().numpy() / np.sqrt(bn.running_var.detach().numpy() + bn.eps)
    return (w * s[:, None, None]).astype(np.float32), ((b - bn.running_mean.detach().numpy()) * s
                                                       + bn.bias.detach().numpy()).astype(np.float32)


def main():
    net, params = sc.load_model()
    d = {"stem_w": net.stem[0].weight.detach().numpy(), "stem_b": net.stem[0].bias.detach().numpy(),
         "head_w": net.head.weight.detach().numpy(), "head_b": net.head.bias.detach().numpy(),
         "n_blocks": np.array(len(net.blocks))}
    for i, blk in enumerate(net.blocks):
        conv1, bn1, _, conv2, bn2 = blk.body
        d[f"b{i}_w1"], d[f"b{i}_b1"] = fold(conv1, bn1)
        d[f"b{i}_w2"], d[f"b{i}_b2"] = fold(conv2, bn2)
        d[f"b{i}_dil"] = np.array(conv1.dilation[0])
    np.savez(OUT, **{k: np.asarray(v) for k, v in d.items()})
    print("wrote", OUT, f"({OUT.stat().st_size // 1024} kB)")

    sys.path.insert(0, str(OUT.parent))
    import stepnet_np
    npnet = stepnet_np.StepNetNP(OUT)
    w = sc.load_recording(sorted((GAIT / "data_ankle").glob("walking_*.csv"))[0])[1][:1500]
    x = sc.features(w)
    with torch.no_grad():
        ref = net(torch.from_numpy(x)[None])[0].numpy()
    got = npnet.forward(x)
    print("max |torch - numpy| over a 15 s walk:", float(np.abs(ref - got).max()))


if __name__ == "__main__":
    main()
