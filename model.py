# model.py
from transformers import MistralForCausalLM
from peft import LoraConfig, get_peft_model
from config import ModelConfig

class Simple1(MistralForCausalLM):
        def __init__(self, config):
                super().__init__(config)
                # Добавляем дополнительные слои для reasoning
                self.reasoning_layers = nn.ModuleList([
                        nn.TransformerDecoderLayer(
                                d_model=config.hidden_size,
                                nhead=config.num_attention_heads
                        ) for _ in range(2)
                ])
                
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
                outputs = super().forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        **kwargs
                )
                
                # Применяем reasoning слои к последним скрытым состояниям
                hidden_states = outputs.hidden_states[-1]
                for layer in self.reasoning_layers:
                        hidden_states = layer(hidden_states, hidden_states)
                        
                logits = self.lm_head(hidden_states)
                return (logits,) + outputs[1:]

def setup_model():
        model = Simple1.from_pretrained(
                ModelConfig.MODEL_NAME,
                load_in_4bit=ModelConfig.USE_4BIT,
                torch_dtype=torch.bfloat16,
                device_map="auto"
        )
        
        # Настройка LoRA
        lora_config = LoraConfig(
                r=ModelConfig.LORA_RANK,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
        )
        
        return get_peft_model(model, lora_config)