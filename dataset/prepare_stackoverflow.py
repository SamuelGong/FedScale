import os

# Albert over StackOverflow
model = "albert-base-v2"
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    AlbertTokenizer,
    MobileBertForPreTraining,
    AutoModelWithLMHead
)
config = AutoConfig.from_pretrained(model)
model = AutoModelWithLMHead.from_config(config)
tokenizer = AlbertTokenizer.from_pretrained(model, do_lower_case=True)

# relative path to this file
root_dir = "data/reddit"
client_data_mapping_dir = os.path.join(root_dir, "client_data_mapping")
train_data_dir = os.path.join(root_dir, "train")
train_mapping_path = os.path.join(client_data_mapping_dir, "train.csv")
test_data_dir = os.path.join(root_dir, "test")
test_mapping_path = os.path.join(client_data_mapping_dir, "test.csv")

def mask_tokens(inputs, tokenizer, args, device='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone().to(device=device)
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability, device=device)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool, device=device), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.tensor(torch.bernoulli(probability_matrix), dtype=torch.bool).detach().to(device=device)
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.tensor(torch.bernoulli(torch.full(labels.shape, 0.8)), dtype=torch.bool, device=device) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.tensor(torch.bernoulli(torch.full(labels.shape, 0.5)), dtype=torch.bool, device=device) & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    bool_indices_random = indices_random
    inputs[bool_indices_random] = random_words[bool_indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

print('here')