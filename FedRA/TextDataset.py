from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import torch


class DebertaTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        """
        Dataset for DeBERTa MLM or classification tasks.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()  # CAUSAL LM needs input_ids as labels
        }


def get_causal_lm_dataset(batch_size=16):
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

    train_texts = dataset['train']['text']
    test_texts = dataset['test']['text']

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    # After loading the tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = DebertaTextDataset(train_texts, tokenizer)
    test_dataset = DebertaTextDataset(test_texts, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader
