from dataclasses import dataclass

@dataclass
class ModelConfig:
        model_name: str = "mistralai/Mistral-7B-v0.1"
        lora_r: int = 16
        lora_alpha: int = 32
        lora_dropout: float = 0.05

@dataclass
class TrainingConfig:
        dataset_name: str = "OpenAssistant/oasst_top1_2023-08-25"
        batch_size: int = 2
        gradient_accumulation: int = 8
        learning_rate: float = 2e-5
        epochs: int = 2
        save_path: str = "./model.safetensors"
        output_dir: str = "./results"
        reasoning_layers: int = 4