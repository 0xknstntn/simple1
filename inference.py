from transformers import AutoTokenizer
from safetensors.torch import load_model
from model import Simple1Model
from config import ModelConfig
import torch

def generate(prompt: str, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(ModelConfig().model_name)
        model = Simple1Model(ModelConfig())
        
        load_model(model, model_path)
        load_model(model, model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
                inputs.input_ids,
                max_length=200,
                temperature=0.7
        )
        return tokenizer.decode(outputs[0])

if __name__ == "__main__":
        import sys
        prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello!"
        print("response: ", generate(prompt, TrainingConfig().save_path))