from transformers import AutoTokenizer
from transformers import TrainerCallback
from datasets import Dataset
from config import ModelConfig
import torch

def save_model(model, path):
        model.save_pretrained(path)

def preprocess_data(dataset: Dataset) -> Dataset:
        model_cfg = ModelConfig()
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        def _format_text(example):
                dialog = []
                for msg in example:
                        dialog.append(msg)
                return {"text": "\n".join(dialog)}

        def _tokenize_fn(examples):
                return tokenizer(
                        examples["text"],
                        max_length=2048,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                        return_attention_mask=True
                )

        if isinstance(dataset, dict):
                dataset = dataset['train']
        
        dataset = dataset.map(_format_text)
        
        columns_to_remove = list(dataset.features.keys())
    
        processed = dataset.map(
                _tokenize_fn,
                batched=True,
                batch_size=1000,
                remove_columns=columns_to_remove,
                num_proc=4
        )
        
        return  processed.train_test_split(test_size=0.1)

class SafeSaveCallback(TrainerCallback):
        def __init__(self, save_path: str):
                self.save_path = save_path
