from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from .common import ROOT, config, environment, now, read_jsonl, write_json
from .modeling import CompletionCollator, encode_example, load_model, resolve_revisions


def train(run_dir, smoke=False):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    cfg = config()
    set_seed(cfg["seed"])
    revisions = resolve_revisions(run_dir)
    model, tokenizer = load_model(cfg["model_id"], revisions[cfg["model_id"]])
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"], target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    model.config.use_cache = False
    model.print_trainable_parameters()
    token_stats = {}
    datasets = {}
    for split in ["train", "validation"]:
        rows = read_jsonl(ROOT / f"data/{split}.jsonl")
        if smoke:
            rows = rows[:8]
        encoded = [encode_example(row, tokenizer, cfg["max_length"]) for row in rows]
        lengths = sorted(len(row["input_ids"]) for row in encoded)
        token_stats[split] = {"n": len(rows), "max": max(lengths), "median": lengths[len(lengths)//2], "supervised_tokens": sum(sum(label != -100 for label in row["labels"]) for row in encoded)}
        datasets[split] = Dataset.from_list(encoded)
    prefix = "smoke_" if smoke else ""
    output = run_dir / (prefix + "checkpoints")
    output.mkdir(parents=True, exist_ok=True)

    class FiniteLoss(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            for key in ("loss", "eval_loss", "grad_norm"):
                if logs and key in logs and not math.isfinite(logs[key]):
                    raise RuntimeError("Неконечный " + key)

    args = TrainingArguments(
        output_dir=str(output), num_train_epochs=cfg["epochs"], max_steps=2 if smoke else -1,
        per_device_train_batch_size=cfg["batch_size"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if smoke else cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"], warmup_ratio=0.03, lr_scheduler_type="cosine",
        fp16=True, bf16=False, optim="paged_adamw_8bit", max_grad_norm=0.3,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps" if smoke else "epoch", eval_steps=2 if smoke else None,
        save_strategy="steps" if smoke else "epoch", save_steps=2 if smoke else 500,
        logging_steps=1 if smoke else 5, save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", seed=cfg["seed"], data_seed=cfg["seed"],
        dataloader_num_workers=0, remove_unused_columns=False, save_safetensors=True,
        eval_accumulation_steps=1, logging_nan_inf_filter=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=datasets["train"], eval_dataset=datasets["validation"], data_collator=CompletionCollator(tokenizer.pad_token_id), processing_class=tokenizer, callbacks=[FiniteLoss()])
    checkpoint = get_last_checkpoint(str(output))
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=checkpoint)
    adapter = run_dir / (prefix + "adapter")
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    trainer.save_state()
    metadata = {"status": "completed", "ended_at": now(), "smoke": smoke, "environment": environment(), "config": cfg, "model_revision": revisions[cfg["model_id"]], "token_stats": token_stats, "metrics": result.metrics, "seconds": time.monotonic()-start, "peak_allocated_gb": torch.cuda.max_memory_allocated()/2**30, "gpu": torch.cuda.get_device_name(0), "best_checkpoint": trainer.state.best_model_checkpoint, "best_eval_loss": trainer.state.best_metric, "log_history": trainer.state.log_history}
    write_json(run_dir / (prefix + "training.json"), metadata)
    (adapter / "README.md").write_text(f"# Houseplants LoRA\n\nBase: {cfg['model_id']}\nRevision: {revisions[cfg['model_id']]}\n\nLoad with PEFT. Base model license applies (qwen-research); see the base model card.\nSynthetic educational dataset, not independently evaluated botanical advice.\n", encoding="utf-8")
    print("Адаптер сохранён:", adapter, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    train(arguments.run_dir, smoke=arguments.smoke)
