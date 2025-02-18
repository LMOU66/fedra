import timm
from .structure import *
from .structuremixer import *

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig


def build_llm_model(model_name="EleutherAI/gpt-neo-125M", task="causal_lm"):
    """
    Builds an LLM with LoRA applied based on the specified task.

    Args:
    - model_name (str): Model identifier on Hugging Face.
    - task (str): Task type - 'causal_lm' or 'sequence_classification'.

    Returns:
    - model: The PEFT-wrapped model.
    - tokenizer: Corresponding tokenizer.
    """

    # Select model class based on task
    if task == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to('cuda')
        lora_task_type = "CAUSAL_LM"
    elif task == "sequence_classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,  # Adjust as needed
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to('cuda')
        lora_task_type = "SEQ_CLS"
    else:
        raise ValueError("Invalid task type. Choose 'causal_lm' or 'sequence_classification'.")

    # Determine target modules dynamically based on model type
    if "gpt-neo" in model_name:
        target_modules = [
            "transformer.h.*.attn.attention.q_proj",
            "transformer.h.*.attn.attention.k_proj",
            "transformer.h.*.attn.attention.v_proj",
            "transformer.h.*.attn.attention.out_proj",
            "transformer.h.*.mlp.c_fc",
            "transformer.h.*.mlp.c_proj"
        ]
    elif "deberta" in model_name:
        target_modules = [
            "deberta.encoder.layer.*.attention.self.query_proj",
            "deberta.encoder.layer.*.attention.self.key_proj",
            "deberta.encoder.layer.*.attention.self.value_proj",
            "deberta.encoder.layer.*.attention.output.dense",
            "deberta.encoder.layer.*.intermediate.dense",
            "deberta.encoder.layer.*.output.dense"
        ]
    else:
        raise ValueError("Model type not supported for LoRA configuration.")

    # Configure LoRA with appropriate task type
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=lora_task_type,
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    # Load tokenizer
    try:
        print(f"🔍 Loading tokenizer for: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # After loading the tokenizer
        tokenizer.pad_token = tokenizer.eos_token

        print("✅ Tokenizer loaded successfully!")
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {e}")
        raise

    return model, tokenizer
