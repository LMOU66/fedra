import copy
import torch
import random
import numpy as np
from tqdm import tqdm
from peft import inject_adapter_in_model, LoraConfig, get_peft_model
from utils import foundationmodel

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Set random seeds for reproducibility
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True


class Server:
    def __init__(self, model, args, A=None, num_layers=12, num_classes=10, clayers=None, depth_cls=0, modeltype='GPT-Neo',
                 lora_config_input=None):
        self.modeltype = modeltype
        self.num_layers = num_layers
        self.client_num = args.domains * args.clients_for_eachdomain


        # Define LoRA configuration for GPT-Neo
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
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

        # Allow external override if provided
        if lora_config_input is not None:
            lora_config = lora_config_input

        # Initialize GPT-Neo model
        from transformers import AutoModelForCausalLM
        base_model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-neo-125M",
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to('cuda')
        self.global_model = get_peft_model(base_model, lora_config).to('cuda')

        # Cache for client parameters
        keys = [k for k in self.global_model.state_dict().keys() if 'lora' in k or 'lm_head' in k or 'norm' in k]
        self.cache_clients = [{k: v for k, v in self.global_model.state_dict().items() if k in keys} for _ in
                              range(self.client_num)]
        self.cache_clients_idx = [{k: False for k in keys} for _ in range(self.client_num)]

        self.client_layers = clayers
        self.exist = np.zeros((self.client_num, self.num_layers), dtype=int)
        self.mlast = {i: [{}, 0] for i in self.client_layers}
        self.lastA = A
        self.A = A

    def agg_ori(self, parameters):
        """Aggregate parameters using a simple mean."""
        globalpara = self.get_para_ori()
        weights = [1 / len(parameters)] * len(parameters)
        for key in parameters[0].keys():
            for c, para in enumerate(parameters):
                if c == 0:
                    globalpara[key] = para[key] * weights[c]
                else:
                    globalpara[key] += para[key] * weights[c]
        self.global_model.load_state_dict(globalpara, strict=False)

    def agg_baseline(self, parameters):
        """Aggregate parameters with LoRA layers only."""
        keys = [k for k in self.global_model.state_dict().keys() if 'lora' in k or 'lm_head' in k or 'norm' in k]
        globalpara = {k: torch.zeros_like(self.global_model.state_dict()[k]) for k in keys}
        num = {k: 0 for k in keys}

        for key in keys:
            for para in parameters:
                tmp = para.get(key, None)
                if tmp is not None:
                    globalpara[key] += tmp
                    num[key] += 1

        for key in keys:
            if num[key] > 0:
                globalpara[key] /= num[key]
            else:
                globalpara[key] = self.global_model.state_dict()[key]

        self.global_model.load_state_dict(globalpara, strict=False)

    def get_para_baseline(self):
        """Get the baseline parameters (LoRA layers only)."""
        keys = [k for k in self.global_model.state_dict().keys() if 'lora' in k or 'lm_head' in k or 'norm' in k]
        return {k: v for k, v in self.global_model.state_dict().items() if k in keys}

    def evaluate(self):
        print("Evaluation logic to be implemented for GPT-Neo.")
