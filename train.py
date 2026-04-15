from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def format_record(record: dict) -> dict:
    prompt = record["prompt"].strip()
    response = record["response"].strip()
    text = f"{prompt}\nAssistant: {response}"
    return {"text": text}


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for ONNX Surgeon action generation.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", default="outputs/onnx-sft")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("Training dependencies are missing. Install transformers, datasets, and accelerate in Colab before running train.py.") from exc

    train_records = [format_record(item) for item in load_records(Path(args.train_file))]
    eval_records = [format_record(item) for item in load_records(Path(args.eval_file))] if args.eval_file else []

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_dataset = Dataset.from_list(train_records).map(tokenize, batched=True, remove_columns=["text"])
    eval_dataset = Dataset.from_list(eval_records).map(tokenize, batched=True, remove_columns=["text"]) if eval_records else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_dataset is not None else "no",
        report_to=[],
        fp16=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
