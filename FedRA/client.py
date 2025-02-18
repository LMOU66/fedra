




import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


set_seed(0)


class KLLoss(nn.Module):
    """KL divergence
    loss for self distillation."""

    def __init__(self):
        super().__init__()
        self.temperature = 1

    def forward(self, pred, label):
        predict = F.log_softmax(pred / self.temperature, dim=1)
        target_data = F.softmax(label / self.temperature, dim=1)
        target_data = target_data + 10 ** (-7)
        with torch.no_grad():
            target = target_data.detach().clone()

        loss = (
                self.temperature
                * self.temperature
                * ((target * (target.log() - predict)).sum(1).sum() / target.size()[0])
        )
        return loss


class Client:
    def __init__(self, dataloader,model_name='EleutherAI/gpt-neo-125M'):
        self.dataloader = dataloader

        # Configure LoRA for GPT-Neo
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "transformer.h.0.attn.attention.q_proj",
                "transformer.h.0.attn.attention.k_proj",
                "transformer.h.0.attn.attention.v_proj",
                "transformer.h.0.attn.attention.out_proj",
                "transformer.h.0.mlp.c_fc",
                "transformer.h.0.mlp.c_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        print(f"🔍 Model name right before loading: {model_name} (type: {type(model_name)})")
        assert isinstance(model_name, str), f"🚨 model_name is not a string! Found: {model_name}"

        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        print("✅ Model loaded successfully!")

        self.local_model = get_peft_model(base_model, lora_config).to('cuda')
        self.last_local_para = None

    def get_para_ori(self):
        return self.local_model.state_dict()

    def get_para(self):
        back = {}
        for k, v in self.local_model.named_parameters():
            if 'lora' in k:
                back[k] = v
        return back

    def load_para(self, para):
        self.local_model.load_state_dict(para, strict=False)

    def train_baseline(self, lr, epochs, mmm):
        self.local_model.train()
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.local_model.parameters()), lr=lr)
        task_criterion = CrossEntropyLoss().to('cuda')

        for epoch in range(epochs):
            for i, batch in enumerate(tqdm(self.dataloader)):
                if i > mmm:
                    break
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to('cuda')
                attention_mask = batch['attention_mask'].to('cuda')
                labels = batch['labels'].to('cuda')

                outputs = self.local_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, :-1]

                loss = task_criterion(logits.reshape(-1, logits.size(-1)), labels.view(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                optimizer.step()

        self.last_local_para = self.local_model.state_dict()

    def train_depthfl(self, lr, epochs, mmm):
        self.local_model.train()
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.local_model.parameters()), lr=lr)
        task_criterion = CrossEntropyLoss().to('cuda')

        for epoch in range(epochs):
            for i, batch in enumerate(tqdm(self.dataloader)):
                if i > mmm:
                    break
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to('cuda')
                attention_mask = batch['attention_mask'].to('cuda')
                labels = batch['labels'].to('cuda')

                outputs = self.local_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, :-1]

                min_size = min(logits.size(1), labels.size(1))
                loss = task_criterion(logits[:, :min_size, :].reshape(-1, logits.size(-1)),
                                      labels[:, :min_size].reshape(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                optimizer.step()

        self.last_local_para = self.local_model.state_dict()
