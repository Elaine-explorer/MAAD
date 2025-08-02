from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, LoraConfig, TaskType
class Detoxifier:
    def __init__(self, model_path, lora_path, tau, temperature, max_new_tokens):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.bfloat16
        )
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=True,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        self.model = PeftModel.from_pretrained(base_model, model_id=lora_path, config=config)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.tau = tau
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def cw_cutoff_generate(self, input_ids, attention_mask):
        generated = input_ids.to(self.device)

        for _ in range(self.max_new_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids=generated, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits / self.temperature, dim=-1)

                max_prob, _ = torch.max(probs, dim=-1, keepdim=True)
                cutoff_mask = probs >= (self.tau * max_prob)
                filtered_probs = probs * cutoff_mask
                filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

                next_token = torch.multinomial(filtered_probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                attention_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        return generated

    def inference(self, prompt, use_cw_cutoff=False):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        if use_cw_cutoff:
            generated_ids = self.cw_cutoff_generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        else:
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=self.temperature,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        output_ids = generated_ids[0, len(inputs["input_ids"][0]):]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)


