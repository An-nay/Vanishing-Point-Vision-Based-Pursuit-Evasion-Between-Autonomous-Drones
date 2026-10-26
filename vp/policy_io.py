from pathlib import Path
import torch


def load_net(path, device="cpu"):
    path = Path(path)
    if path.suffix != ".pt" or not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    algo = torch.load(path, map_location=device, weights_only=True).get("algo", "maddpg")
    if algo == "mappo":
        from vp.mappo import MAPPO
        return MAPPO.load(path, device)
    from vp.maddpg import MADDPG
    return MADDPG.load(path, device)
