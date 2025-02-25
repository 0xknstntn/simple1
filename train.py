from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from safetensors.torch import save_model
from model import Simple1Model
from config import ModelConfig, TrainingConfig
from utils import preprocess_data, SafeSaveCallback
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()
login(os.getenv('HF_KEY'))

def train():
        model_cfg = ModelConfig()
        train_cfg = TrainingConfig()
        
        dataset = load_dataset(train_cfg.dataset_name)
        dataset = preprocess_data(dataset)
        
        model = Simple1Model(model_cfg)
    
        training_args = TrainingArguments(
                output_dir=train_cfg.output_dir,
                per_device_train_batch_size=train_cfg.batch_size,
                gradient_accumulation_steps=train_cfg.gradient_accumulation,
                learning_rate=train_cfg.learning_rate,
                num_train_epochs=train_cfg.epochs,
                fp16=True,
                logging_steps=50,
                save_strategy="epoch"
        )
    
        trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                callbacks=[SafeSaveCallback(train_cfg.save_path)]
        )
        trainer.train()
    
        model.save_pretrained(train_cfg.output_dir)

if __name__ == "__main__":
        train()