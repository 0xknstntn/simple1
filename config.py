# config.py
from transformers import MistralConfig

class ModelConfig:
    MODEL_NAME = "mistralai/Mistral-7B-v0.1"
    DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
    MAX_LENGTH = 2048  # Увеличиваем контекстное окно
    BATCH_SIZE = 2      # Уменьшаем из-за памяти
    GRAD_ACCUM_STEPS = 16
    LEARNING_RATE = 1e-5
    EPOCHS = 2
    SAVE_DIR = "simple1"
    LORA_RANK = 64      # Для LoRA адаптера
    USE_4BIT = True     # 4-битная квантизация
    USE_FLASH_ATTN = True # Flash Attention