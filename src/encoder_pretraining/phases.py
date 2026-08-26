"""Token/step-budgeted, checkpointed objective phases."""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Phase:
    objective: str; steps: int | None = None; tokens: int | None = None
    def __post_init__(self):
        if (self.steps is None)==(self.tokens is None): raise ValueError("Specify exactly one of steps or tokens")

class PhaseRunner:
    def __init__(self, model, tokenizer, output_dir): self.model,self.tokenizer,self.output_dir=model,tokenizer,Path(output_dir)
    def transition(self, phase_index, objective):
        from .objectives import set_attention_mode
        if phase_index:
            target=self.output_dir/f"phase-{phase_index}-boundary"; target.mkdir(parents=True,exist_ok=True)
            self.model.save_pretrained(target); self.tokenizer.save_pretrained(target)
        set_attention_mode(self.model,objective)
