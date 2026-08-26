"""Complete checkpoints: weights plus resumable optimizer/RNG state."""
from pathlib import Path
import random

def save_checkpoint(path, model, tokenizer, optimizer, scheduler, state):
    import torch
    path = Path(path); model_dir = path / "model"; model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir); tokenizer.save_pretrained(model_dir)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "state": state,
                "torch_rng": torch.get_rng_state(), "python_rng": random.getstate()}, path / "training_state.pt")

def load_training_state(path, optimizer, scheduler):
    import torch
    payload = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(payload["optimizer"]); scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng"]); random.setstate(payload["python_rng"])
    return payload["state"]
