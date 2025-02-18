import math
import torch
import torch.nn as nn
from tqdm import tqdm
from models.GetModel import build_llm_model
from peft import get_peft_model
from transformers import AutoModelForCausalLM

def lrcos(step=0, lr=0.01, lr_min=0.0001, T_max=500):
    """Cosine learning-rate scheduler."""
    return 0.5 * (1 + math.cos(math.pi * step / T_max)) * (lr - lr_min) + lr_min


class foundationmodel(nn.Module):
    def __init__(self, layer=12, num_classes=10, depth_cls=0, modeltype='GPT-Neo', lora_config=None):
        super(foundationmodel, self).__init__()

        # Build the base model with GPT-Neo
        if modeltype == 'GPT-Neo':
            self.back = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

            # Apply LoRA if config provided
            if lora_config:
                self.back = get_peft_model(self.back, lora_config)

        else:
            # Fallback to other model types if needed
            self.back = build_llm_model(num_classes=num_classes,
                                          edge_size=224,
                                          modeltype=modeltype,
                                          patch_size=16,
                                          Prompt_Token_num=0,
                                          depth=layer,
                                          depth_cls=depth_cls)

            if lora_config:
                self.back = get_peft_model(self.back, lora_config)

    def forward(self, x):
        return self.back(x)


def evaluation_depthfl(model, testdata):
    """Evaluate a model that outputs multiple intermediate layer predictions (depthFL approach)."""
    model.eval()
    total_loss, total_samples = 0, 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(testdata, desc="Evaluating GPT-Neo"):
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return ppl


def evaluation(model, testdata):
    """Evaluate model performance for a classification task."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in tqdm(testdata, desc="Evaluating GPT-Neo"):
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def evaluation_llm(model, test_loader):
    """Evaluate GPT-Neo model on a causal language modeling task using perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating GPT-Neo LLM"):
            input_ids = batch["input_ids"].to('cuda')
            attention_mask = batch["attention_mask"].to('cuda')
            labels = batch["labels"].to('cuda')

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return ppl
