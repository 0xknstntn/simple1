from transformers import Trainer, TrainingArguments
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import save_model
from model import Simple1Model
from config import ModelConfig, TrainingConfig
from utils import preprocess_data, SafeSaveCallback
from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch

load_dotenv()
login(os.getenv('HF_KEY'))

def data_collator(data):
        # Convert lists to tensors
        for x in data:
                if isinstance(x['input_ids'], list):
                        x['input_ids'] = torch.tensor(x['input_ids'])
                if isinstance(x['attention_mask'], list):
                        x['attention_mask'] = torch.tensor(x['attention_mask'])

        return {
                'input_ids': torch.stack([x['input_ids'] for x in data]),
                'attention_mask': torch.stack([x['attention_mask'] for x in data]),
                'labels': torch.stack([x['input_ids'] for x in data])
        }

class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                # Forward pass
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs

                # Compute loss
                loss = self.custom_compute_loss(logits, labels)

                return (loss, outputs) if return_outputs else loss

        def custom_compute_loss(self, model_output, target):
                # Reshape model_output to (batch_size * seq_len, vocab_size)
                model_output_reshaped = model_output.view(-1, model_output.size(-1))

                # Reshape target to (batch_size * seq_len)
                target_reshaped = target.view(-1)

                # Compute the loss
                loss = F.cross_entropy(model_output_reshaped, target_reshaped)

                return loss


def train():
        model_cfg = ModelConfig()
        train_cfg = TrainingConfig()
        
        dataset = load_dataset(train_cfg.dataset_name)
        dataset = preprocess_data(dataset)

        print(dataset)
        if isinstance(dataset, dict):
                dataset = dataset['train']
        
        model = Simple1Model(model_cfg)
    
        training_args = TrainingArguments(
                output_dir=train_cfg.output_dir,
                per_device_train_batch_size=train_cfg.batch_size,
                gradient_accumulation_steps=train_cfg.gradient_accumulation,
                learning_rate=train_cfg.learning_rate,
                num_train_epochs=train_cfg.epochs,
                fp16=True,
                logging_steps=10,
                #save_strategy="steps",
                remove_unused_columns=False
        )
    
        trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                callbacks=[SafeSaveCallback(train_cfg.save_path)],
                data_collator=data_collator
        )
        trainer.train()
    
        trainer.save_model(train_cfg.output_dir)

if __name__ == "__main__":
        train()
