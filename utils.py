from transformers import AutoTokenizer
from transformers import TrainerCallback
from datasets import Dataset
from config import ModelConfig
import torch

def preprocess_data(dataset: Dataset) -> Dataset:
        """Подготовка диалоговых данных для обучения"""
        model_cfg = ModelConfig()
        
        # Загрузка токенизатора
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        def _combine_messages(example):
                """Объединение сообщений диалога в единый текст"""
                return {
                        "text": "\n".join(
                                [f"{m['role']}: {m['content']}" for m in example['messages']]
                        )
                }

        def _tokenize_fn(examples):
                """Токенизация с обработкой последовательностей"""
                # Объединение диалогов
                texts = [_combine_messages(ex)['text'] for ex in examples]
                
                # Токенизация с учетом максимальной длины
                tokenized = tokenizer(
                        texts,
                        max_length=512,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                        return_attention_mask=True
                )
                
                # Создание labels для языкового моделирования
                tokenized["labels"] = tokenized["input_ids"].clone()
                
                # Конвертация в numpy для совместимости с Dataset
                return {k: v.numpy() for k, v in tokenized.items()}

        # Применение обработки
        processed = dataset.map(
                _tokenize_fn,
                batched=True,
                batch_size=1000,
                remove_columns=dataset.column_names,
                num_proc=4
        )

        # Фильтрация пустых примеров
        processed = processed.filter(
                lambda ex: ex["input_ids"].any(),
                num_proc=4
        )
        
        return processed.train_test_split(test_size=0.1)

class SafeSaveCallback(TrainerCallback):
        def __init__(self, save_path: str):
                self.save_path = save_path
        
        def on_epoch_end(self, args, state, control, **kwargs):
                save_model(kwargs['model'], self.save_path)