import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class GPTNeoPromptModel(nn.Module):
    def __init__(self, model_name="EleutherAI/gpt-neo-125M", prompt_token_num=1):
        super(GPTNeoPromptModel, self).__init__()

        # Load GPT-Neo as a Causal LM
        self.gptneo = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # Initialize prompt tokens
        hidden_size = self.gptneo.config.hidden_size
        self.prompt_tokens = nn.Parameter(torch.randn(1, prompt_token_num, hidden_size))

    def freeze_base_model(self):
        """Freeze all parameters except prompt tokens."""
        for param in self.gptneo.parameters():
            param.requires_grad = False
        self.prompt_tokens.requires_grad = True

    def unfreeze_base_model(self):
        """Unfreeze the entire model."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass with prompt token injection.
        """
        batch_size = input_ids.size(0)
        # Inject prompt tokens
        prompt_tokens = self.prompt_tokens.expand(batch_size, -1, -1)

        # Concatenate prompt tokens with input embeddings
        inputs_embeds = self.gptneo.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prompt_tokens, inputs_embeds], dim=1)

        # Adjust attention mask accordingly
        if attention_mask is not None:
            prompt_attention = torch.ones((batch_size, self.prompt_tokens.size(1)), device=attention_mask.device)
            attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)

        # Pass through GPT-Neo
        outputs = self.gptneo(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        # Return logits for Causal LM tasks
        return outputs.logits
