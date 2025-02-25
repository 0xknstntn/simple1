from transformers import AutoTokenizer
from transformers import TrainerCallback
from datasets import Dataset
from config import ModelConfig
import torch

def preprocess_data(dataset: Dataset) -> Dataset:
        """Подготовка диалоговых данных для обучения"""
        model_cfg = ModelConfig()
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        def _format_text(example):
                """Полноценное форматирование диалога"""
                dialog = []
                for msg in example['message_tree']:
                        if msg['parent'] is None:  # Начало диалога
                                dialog.append(f"User: {msg['content']}")
                        else:
                                role = 'Assistant' if msg['role'] == 'bot' else 'User'
                                dialog.append(f"{role}: {msg['content']}")
                return {"text": "\n".join(dialog)}

        def _tokenize_fn(examples):
                """Токенизация с обработкой последовательностей"""
                texts = [ex['text'] for ex in examples]
                tokenized = tokenizer(
                        texts,
                        max_length=512,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                        return_attention_mask=True
                )
                return {
                        "input_ids": tokenized["input_ids"].squeeze(),
                        "attention_mask": tokenized["attention_mask"].squeeze(),
                        "labels": tokenized["input_ids"].squeeze().clone()
                }

        # Если dataset в формате DatasetDict (train/test split)
        if isinstance(dataset, dict):
                dataset = dataset['train']
        
        # Применяем преобразования
        dataset = dataset.map(_format_text)
        
        # Получаем список реальных колонок после преобразований
        columns_to_remove = list(dataset.features.keys())
    
        # Токенизация и удаление исходных колонок
        processed = dataset.map(
                _tokenize_fn,
                batched=True,
                batch_size=1000,
                remove_columns=columns_to_remove,
                num_proc=4
        )
        
        # Фильтрация пустых примеров
        processed = processed.filter(
                lambda ex: any(ex["input_ids"]),
                num_proc=4
        )
        
        return processed.train_test_split(test_size=0.1)

class SafeSaveCallback(TrainerCallback):
        def __init__(self, save_path: str):
                self.save_path = save_path
        
        def on_epoch_end(self, args, state, control, **kwargs):
                save_model(kwargs['model'], self.save_path)