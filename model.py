import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from config import ModelConfig

class Simple1Model(nn.Module):
        def __init__(self, cfg: ModelConfig):
                super().__init__()
                self.base_model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
                self._add_reasoning_layers()
                self._setup_lora(cfg)
    
        def _add_reasoning_layers(self):
                # Добавление reasoning блоков
                hidden_size = self.base_model.config.hidden_size
                self.reasoning = nn.TransformerEncoder(
                        encoder_layer=nn.TransformerEncoderLayer(
                                d_model=hidden_size,
                                nhead=self.base_model.config.num_attention_heads
                        ),
                        num_layers=2
                )
    
        def _setup_lora(self, cfg: ModelConfig):
                # Настройка LoRA
                lora_config = LoraConfig(
                        r=cfg.lora_r,
                        lora_alpha=cfg.lora_alpha,
                        target_modules=["q_proj", "v_proj"],
                        lora_dropout=cfg.lora_dropout,
                        task_type="CAUSAL_LM"
                )
                self.base_model = get_peft_model(self.base_model, lora_config)
    
        def forward(self, inputs):
                # Реализация forward pass
                outputs = self.base_model(**inputs, output_hidden_states=True)
                processed = self.reasoning(outputs.hidden_states[-1])
                return self.base_model.lm_head(processed)