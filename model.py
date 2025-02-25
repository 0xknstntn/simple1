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
                #self.num_layers = cfg.reasoning_layers

        def _add_reasoning_layers(self):
                hidden_size = self.base_model.config.hidden_size
                self.reasoning = nn.TransformerEncoder(
                        encoder_layer=nn.TransformerEncoderLayer(
                                d_model=hidden_size,
                                nhead=self.base_model.config.num_attention_heads
                        ),
                        num_layers=4 #self.num_layers
                )
    
        def _setup_lora(self, cfg: ModelConfig):
                lora_config = LoraConfig(
                        r=cfg.lora_r,
                        lora_alpha=cfg.lora_alpha,
                        target_modules=["q_proj", "v_proj"],
                        lora_dropout=cfg.lora_dropout,
                        task_type="CAUSAL_LM"
                )
                self.base_model = get_peft_model(self.base_model, lora_config)
    
        def forward(self, input_ids, attention_mask=None, **kwargs):
                inputs = {'input_ids': input_ids}
                if attention_mask is not None:
                        inputs['attention_mask'] = attention_mask

                outputs = self.base_model(**inputs, output_hidden_states=True, **kwargs)
                processed = self.reasoning(outputs.hidden_states[-1])
                final_output = self.base_model.lm_head(processed)

                #print("Final output shape:", final_output.shape)

                return final_output
