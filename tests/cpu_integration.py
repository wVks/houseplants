"""Два шага маленькой случайной Qwen2 + LoRA. Не результат целевого обучения."""
from __future__ import annotations

import tempfile
from pathlib import Path

from src.common import ROOT, environment, now, read_jsonl, sha256, write_json
from src.modeling import CompletionCollator, encode_example


def run():
    import torch
    from datasets import Dataset, load_dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, Trainer, TrainingArguments, set_seed
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    set_seed(42)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / ".cache/qwen-tokenizer", local_files_only=True)
    token_stats = {}
    for split in ["train", "validation", "test"]:
        rows = read_jsonl(ROOT / f"data/{split}.jsonl")
        encoded = [encode_example(row, tokenizer, 1024) for row in rows]
        token_stats[split] = {"examples": len(rows), "max_tokens": max(len(row["input_ids"]) for row in encoded), "min_assistant_tokens": min(sum(label != -100 for label in row["labels"]) for row in encoded)}
        loaded = load_dataset("json", data_files=str(ROOT / f"data/{split}.jsonl"), split="train", cache_dir=str(ROOT / ".cache/datasets"))
        assert len(loaded) == len(rows)
    write_json(ROOT / "results/tokenization_audit.json", {"tokenizer_files_sha256": {path.name: sha256(path) for path in (ROOT / ".cache/qwen-tokenizer").glob("*.json")}, "splits": token_stats})
    # Маленький словарь не требует загрузки весов 3B для проверки Trainer/PEFT.
    toy_tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}, unk_token="<unk>")), pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    model_config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128, pad_token_id=0, eos_token_id=3, attention_dropout=0.0)
    base = Qwen2ForCausalLM(model_config)
    base_state = {key: value.detach().clone() for key, value in base.state_dict().items()}
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM"))
    samples = [{"input_ids": [2, 4, 5, 7, 3], "attention_mask": [1]*5, "labels": [-100, -100, -100, 7, 3]}, {"input_ids": [2, 5, 8, 3], "attention_mask": [1]*4, "labels": [-100, -100, 8, 3]}]
    collator = CompletionCollator(0)
    batch = collator(samples)
    assert batch["labels"][1, -1].item() == -100
    assert batch["labels"][0, -1].item() == 3
    before = {key: value.detach().clone() for key, value in model.named_parameters() if value.requires_grad}
    with tempfile.TemporaryDirectory() as directory:
        arguments = TrainingArguments(output_dir=directory, use_cpu=True, max_steps=2, per_device_train_batch_size=2, learning_rate=0.01, logging_steps=1, save_strategy="no", report_to="none", dataloader_pin_memory=False)
        trainer = Trainer(model=model, args=arguments, train_dataset=Dataset.from_list(samples), data_collator=collator, processing_class=toy_tokenizer)
        result = trainer.train()
        assert torch.isfinite(torch.tensor(result.training_loss))
        assert any(not torch.equal(before[key], value) for key, value in model.named_parameters() if key in before)
        model.eval()
        with torch.no_grad():
            expected = model(**batch).logits
        adapter = Path(directory) / "adapter"
        model.save_pretrained(adapter, safe_serialization=True)
        assert (adapter / "adapter_model.safetensors").is_file()
        reloaded_base = Qwen2ForCausalLM(model_config)
        reloaded_base.load_state_dict(base_state)
        reloaded = PeftModel.from_pretrained(reloaded_base, adapter).eval()
        with torch.no_grad():
            actual = reloaded(**batch).logits
        torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-5)
    write_json(ROOT / "results/local_validation.json", {"status": "passed", "at": now(), "environment": environment(), "checks": ["real_qwen_tokenizer_all_examples_under_1024", "datasets_load_dataset_all_splits", "prompt_labels_masked", "eos_labels_kept", "padding_labels_masked", "cpu_tiny_qwen_lora_two_steps", "trainable_parameters_changed", "safetensors_saved", "adapter_reload_logits_match"], "scope": "Случайная маленькая CPU-модель; целевые веса 3B не обучались, T4 и bitsandbytes не проверены."})
    print("CPU integration: PASSED", token_stats)


if __name__ == "__main__":
    run()
