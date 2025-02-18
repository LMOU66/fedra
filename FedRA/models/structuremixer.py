import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class GPTNeoMixer(nn.Module):

    def __init__(
            self,
            model_name="EleutherAI/gpt-neo-125M",
            depth_cls=0,
            prompt_token_num=1
    ):
        super().__init__()
        self.depth_cls = depth_cls

        # Load GPT-Neo model for Causal Language Modeling
        self.gptneo = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # Initialize prompt tokens
        hidden_size = self.gptneo.config.hidden_size
        self.prompt_tokens = nn.Parameter(torch.randn(1, prompt_token_num, hidden_size))

        # Optional: Add intermediate layer heads
        if self.depth_cls > 0:
            self.mid_heads = nn.ModuleList([
                nn.Linear(hidden_size, self.gptneo.config.vocab_size)
                for _ in range(self.depth_cls)
            ])

    def freeze_base_model(self):
        """Freeze all parameters except the prompt tokens."""
        for param in self.gptneo.parameters():
            param.requires_grad = False
        self.prompt_tokens.requires_grad = True

    def unfreeze_base_model(self):
        """Unfreeze the entire model."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, return_intermediate=False):
        """
        Forward pass with optional intermediate outputs.
        """
        batch_size = input_ids.size(0)

        # Inject prompt tokens
        prompt_tokens = self.prompt_tokens.expand(batch_size, -1, -1)

        # Convert input IDs to embeddings
        inputs_embeds = self.gptneo.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prompt_tokens, inputs_embeds], dim=1)

        # Adjust attention mask for prompt tokens
        if attention_mask is not None:
            prompt_attention = torch.ones((batch_size, self.prompt_tokens.size(1)), device=attention_mask.device)
            attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)

        # Forward pass through GPT-Neo
        outputs = self.gptneo(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)

        # Extract hidden states for intermediate outputs
        hidden_states = outputs.hidden_states

        # Generate intermediate predictions if depth_cls > 0
        if self.depth_cls > 0 and return_intermediate:
            intermediate_outputs = [self.mid_heads[i](hidden_states[i][:, 0]) for i in range(self.depth_cls)]
            return intermediate_outputs + [outputs.logits]

        return outputs.logits
