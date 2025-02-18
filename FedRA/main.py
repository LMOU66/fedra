
import logging
logging.basicConfig(level=logging.DEBUG)

# Print Huggingface cache location
import os
print(f"Huggingface cache directory: {os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))}")


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"




import torch
import random
import logging
from tqdm import tqdm
import numpy as np
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model
from client import Client
from server import Server
import os
from transformers import AutoModelForCausalLM
from TextDataset import get_causal_lm_dataset



# Define the path for the logs directory
log_dir = os.path.join(os.getcwd(), 'finallogs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', default="./wikitext_dataset")
    parser.add_argument('--rounds', default=10, type=int, help='Number of training rounds')
    parser.add_argument('--clients', default=5, type=int, help='Number of clients')
    parser.add_argument('--model_name', default="EleutherAI/gpt-neo-125M", help='Model name')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='Learning rate')
    parser.add_argument('--client_epochs', default=3, type=int, help='Epochs per client per round')
    parser.add_argument('--data', default='qwen', help='Dataset type')
    parser.add_argument('--modeltype', default='llm', help='Model type')
    parser.add_argument('--domains', default=1, type=int, help='Number of domains')
    parser.add_argument('--clients_for_eachdomain', default=1, type=int, help='Number of clients per domain')
    return parser


args = get_parser().parse_args()
train_loader, _ = get_causal_lm_dataset(batch_size=args.batch_size)

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

# Logging
logging.basicConfig(
    filename=f'./finallogs/federated_llm_training.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w'
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)



# Model loading for GPT-Neo-125M
base_model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)



# Configure LoRA
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
clayers = {
    0: [0],    # Client 0 updates only the first LoRA layer
    1: [1],    # Client 1 updates only the second LoRA layer
    2: [2],    # Client 2 updates only the third LoRA layer
}


# Wrap the base model with LoRA
global_model = get_peft_model(base_model, lora_config)
global_model.train()
print(f"🔍 Model name passed to Client: {args.model_name} (type: {type(args.model_name)})")
# Initialize clients
clients = [Client(dataloader=train_loader, model_name=args.model_name) for _ in range(args.clients)]
server = Server(global_model, args,clayers=clayers)

# Federated training loop
for round_idx in tqdm(range(args.rounds), desc="Federated Training Rounds"):
    logger.info(f"Round {round_idx+1}/{args.rounds}")
    client_params = []

    for client_idx, client in enumerate(clients):
        logger.info(f"Training client {client_idx+1}")
        client.train_depthfl(args.learning_rate, args.client_epochs, 100)
        client_params.append(client.get_parameters())

    server.aggregate(client_params)

    # Evaluate global model performance (optional)
    if round_idx % 5 == 0 or round_idx == args.rounds - 1:
        logger.info("Evaluating global model")
        server.evaluate()

logger.info("Federated training completed.")


# server.py code

class Server:
    def __init__(self, model, num_clients):
        self.model = model
        self.num_clients = num_clients

    def aggregate(self, client_params):
        with torch.no_grad():
            avg_params = {}
            for param_name in client_params[0].keys():
                avg_params[param_name] = torch.mean(torch.stack([cp[param_name] for cp in client_params]), dim=0)
            self.model.load_state_dict(avg_params, strict=False)

    def evaluate(self):
        print("Evaluation logic to be implemented.")
