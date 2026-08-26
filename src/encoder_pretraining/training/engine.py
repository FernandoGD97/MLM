"""Objective-neutral epoch loop with epoch-zero and main-process-only evaluation."""
from dataclasses import dataclass
import math
from pathlib import Path
from .optimization import build_optimizer, build_scheduler
from .checkpointing import save_checkpoint

@dataclass
class TargetWeightedAccumulator:
    nll_sum: float = 0.0
    targets: int = 0
    processed_tokens: int = 0
    def update(self, mean_loss, targets, processed_tokens):
        self.nll_sum += float(mean_loss) * int(targets); self.targets += int(targets); self.processed_tokens += int(processed_tokens)
    @property
    def nll(self): return self.nll_sum / self.targets if self.targets else float("nan")

class TrainingEngine:
    def __init__(self, model, tokenizer, train_dataset, collator, objective, config, evaluator, tracker, accelerator=None):
        from accelerate import Accelerator
        import torch
        self.accelerator = accelerator or Accelerator(gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)), mixed_precision=config.get("mixed_precision", "no"))
        self.model, self.tokenizer, self.objective, self.config, self.collator = model, tokenizer, objective, config, collator
        self.evaluator, self.tracker = evaluator, tracker
        generator=torch.Generator().manual_seed(int(config.get("seed",42)))
        self.loader=torch.utils.data.DataLoader(train_dataset,batch_size=int(config["batch_size"]),shuffle=True,collate_fn=collator,generator=generator)
        self.optimizer=build_optimizer(model,config)
        updates=math.ceil(len(self.loader)/int(config.get("gradient_accumulation_steps",1)))*int(config["epochs"])
        self.scheduler=build_scheduler(self.optimizer,config,updates)
        self.model,self.optimizer,self.loader,self.scheduler=self.accelerator.prepare(model,self.optimizer,self.loader,self.scheduler)
        self.run_dir=Path(config["run_dir"]); self.global_step=0; self.processed_tokens=0; self.start_epoch=0
        self.token_budget=sum(len(x["input_ids"]) for x in train_dataset)*int(config["epochs"])
        if config.get("resume_from_checkpoint"):
            from .checkpointing import load_training_state
            state=load_training_state(config["resume_from_checkpoint"],self.optimizer,self.scheduler)
            self.global_step=int(state["global_step"]); self.processed_tokens=int(state["processed_tokens"]); self.start_epoch=int(state["epoch"])

    def run(self):
        self.run_dir.mkdir(parents=True,exist_ok=True)
        if self.config.get("run_before_training",True) and not self.start_epoch: self._evaluate(0)
        for epoch in range(self.start_epoch+1,int(self.config["epochs"])+1):
            train_metrics=self._train_epoch()
            self.accelerator.wait_for_everyone()
            checkpoint=self.run_dir/f"epoch_{epoch:03d}"
            if self.accelerator.is_main_process:
                save_checkpoint(checkpoint,self.accelerator.unwrap_model(self.model),self.tokenizer,self.optimizer,self.scheduler,
                                {"epoch":epoch,"global_step":self.global_step,"processed_tokens":self.processed_tokens})
                self.tracker.add_metrics(epoch,self.global_step,self.processed_tokens,self.objective,"training","train",train_metrics)
            self._evaluate(epoch,checkpoint)
        return self.run_dir

    def _train_epoch(self):
        import torch
        self.model.train(); acc=TargetWeightedAccumulator(); optimizer_steps=0
        for batch in self.loader:
            labels=batch["labels"]; targets=int((labels!=-100).sum().item()); processed=int(batch["attention_mask"].sum().item())
            with self.accelerator.accumulate(self.model):
                output=self.model(**batch); loss=output.loss
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.optimizer.step(); self.scheduler.step(); self.optimizer.zero_grad(); self.global_step+=1; optimizer_steps+=1
            acc.update(loss.detach().float().item(),targets,processed); self.processed_tokens+=processed
            if hasattr(self.collator,"set_progress"): self.collator.set_progress(self.processed_tokens,self.token_budget)
        totals=torch.tensor([acc.nll_sum,acc.targets,acc.processed_tokens],device=self.accelerator.device,dtype=torch.float64)
        totals=self.accelerator.reduce(totals,reduction="sum")
        acc.nll_sum=float(totals[0]); acc.targets=int(totals[1]); acc.processed_tokens=int(totals[2])
        prefix="train_clm" if self.objective=="clm" else "train_mlm"
        metrics={f"{prefix}_nll":acc.nll,"processed_tokens":acc.processed_tokens,"optimizer_steps":optimizer_steps,
                 "learning_rate":self.scheduler.get_last_lr()[0]}
        if self.objective=="clm": metrics.update(train_clm_ppl=math.exp(min(acc.nll,50)),clm_prediction_targets=acc.targets)
        else: metrics.update(train_masked_targets=acc.targets,effective_mask_ratio=acc.targets/max(1,acc.processed_tokens))
        return metrics

    def _evaluate(self,epoch,checkpoint=None):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.evaluator.evaluate(self.accelerator.unwrap_model(self.model),self.tokenizer,checkpoint or self.run_dir/"epoch_000",
                                    epoch,self.global_step,self.processed_tokens)
        self.accelerator.wait_for_everyone()
