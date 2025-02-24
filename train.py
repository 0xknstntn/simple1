# train.py
from transformers import TrainingArguments, Trainer
from model import setup_model
from data_processing import process_data
from config import ModelConfig
import torch

def train():
        dataset, tokenizer = process_data()
        model = setup_model()
        
        training_args = TrainingArguments(
                output_dir=ModelConfig.SAVE_DIR,
                per_device_train_batch_size=ModelConfig.BATCH_SIZE,
                gradient_accumulation_steps=ModelConfig.GRAD_ACCUM_STEPS,
                learning_rate=ModelConfig.LEARNING_RATE,
                num_train_epochs=ModelConfig.EPOCHS,
                logging_steps=10,
                optim="paged_adamw_32bit",
                save_strategy="steps",
                save_steps=500,
                bf16=torch.cuda.is_bf16_supported(),
                fp16=not torch.cuda.is_bf16_supported(),
                gradient_checkpointing=True,
                report_to="none"
        )
        
        trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=lambda data: {
                        'input_ids': torch.stack([x['input_ids'] for x in data]),
                        'attention_mask': torch.stack([x['attention_mask'] for x in data]),
                        'labels': torch.stack([x['input_ids'] for x in data])
                }
        )
        
        trainer.train()
        model.save_pretrained(ModelConfig.SAVE_DIR)