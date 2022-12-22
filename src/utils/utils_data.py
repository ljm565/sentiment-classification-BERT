import torch
from torch.utils.data import Dataset



class DLoader(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = config.max_len
        self.length = len(self.data)


    def encoding(self, s):
        results = self.tokenizer.tokenizer.encode_plus(
            text=s,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return results['input_ids'][0], results['attention_mask'][0].type(torch.FloatTensor)


    def __getitem__(self, idx):
        s, label = self.data[idx][0], self.data[idx][1]
        s, attn_mask = self.encoding(s)
        return s, torch.tensor(label), attn_mask 

    
    def __len__(self):
        return self.length