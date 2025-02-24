# inference.py
from transformers import AutoTokenizer, pipeline
from peft import PeftModel
from config import ModelConfig
import torch

class MistralAssistant:
        def __init__(self):
                self.tokenizer = AutoTokenizer.from_pretrained(ModelConfig.MODEL_NAME)
                base_model = AutoModelForCausalLM.from_pretrained(
                        ModelConfig.MODEL_NAME,
                        load_in_4bit=True,
                        torch_dtype=torch.bfloat16,
                        device_map="auto"
                )
                self.model = PeftModel.from_pretrained(
                        base_model,
                        ModelConfig.SAVE_DIR
                )
                self.pipe = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device_map="auto",
                )
                
        def generate(self, prompt):
                messages = [
                        {"role": "user", "content": prompt}
                ]
                
                prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                )
                
                outputs = self.pipe(
                        prompt,
                        max_new_tokens=512,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                )
                
                return outputs[0]['generated_text'].split("<|assistant|>")[-1].strip()