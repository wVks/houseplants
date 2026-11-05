from __future__ import annotations

from .common import config, read_json, write_json


def resolve_revisions(run_dir):
    """Один commit для каждого репозитория на весь запуск, включая возобновление."""
    from huggingface_hub import HfApi
    path = run_dir / "model_revisions.json"
    if path.exists():
        return read_json(path)
    cfg = config()
    api = HfApi()
    revisions = {model_id: api.model_info(model_id).sha for model_id in [cfg["model_id"], cfg["judge_model_id"]]}
    write_json(path, revisions)
    return revisions


def load_model(model_id, revision, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("Нужна NVIDIA GPU. В Colab выберите Runtime → Change runtime type → T4 GPU.")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
        device_map={"": 0}, attn_implementation="sdpa",
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    return model, tokenizer


def encode_example(row, tokenizer, max_length):
    """Loss только на assistant, включая im_end. Тихое обрезание запрещено."""
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full[:len(prompt)] != prompt:
        raise ValueError("Chat template изменяет префикс при добавлении ответа: " + row["id"])
    if len(full) > max_length:
        raise ValueError(f"{row['id']}: {len(full)} токенов > {max_length}; сократите пример или явно измените max_length.")
    labels = [-100] * len(prompt) + full[len(prompt):]
    if tokenizer.eos_token_id not in labels[len(prompt):]:
        raise ValueError("Ответ не содержит токен завершения: " + row["id"])
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}


class CompletionCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        width = max(len(row["input_ids"]) for row in features)
        result = {key: [] for key in ("input_ids", "attention_mask", "labels")}
        for row in features:
            padding = width - len(row["input_ids"])
            result["input_ids"].append(row["input_ids"] + [self.pad_token_id] * padding)
            result["attention_mask"].append(row["attention_mask"] + [0] * padding)
            result["labels"].append(row["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def generate_one(model, tokenizer, messages, max_new_tokens):
    import torch
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    model.eval()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
    tokens = generated[0, inputs["input_ids"].shape[1]:].tolist()
    ended = bool(tokens) and tokens[-1] == tokenizer.eos_token_id
    return {"answer": tokenizer.decode(tokens, skip_special_tokens=True), "output_tokens": len(tokens), "input_tokens": inputs["input_ids"].shape[1], "truncated": not ended and len(tokens) >= max_new_tokens}
