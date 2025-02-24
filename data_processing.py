# data_processing.py
from transformers import AutoTokenizer
from datasets import load_dataset
from config import ModelConfig

def format_chatml(example):
        formatted = []
        for msg in example['messages']:
                if msg['role'] == 'system':
                        formatted.append(f"<|im_start|>system\n{msg['content']}<|im_end|>")
                else:
                        formatted.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        return {"text": "\n".join(formatted)}

def process_data():
        tokenizer = AutoTokenizer.from_pretrained(ModelConfig.MODEL_NAME)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        dataset = load_dataset(ModelConfig.DATASET_NAME, split='train_sft')
        dataset = dataset.map(format_chatml, batched=False)
        
        def tokenize_fn(examples):
                return tokenizer(
                        examples['text'],
                        truncation=True,
                        max_length=ModelConfig.MAX_LENGTH,
                        padding='max_length',
                        add_special_tokens=False
                )
                
        return dataset.map(tokenize_fn, batched=True), tokenizer