from transformers import BertModel

import torch
import torch.nn as nn

from tools.tokenizers import BERTTokenizer


# BERT
class BERT(nn.Module):
    def __init__(self, config, device):
        super(BERT, self).__init__()
        self.pretrained_model = config.pretrained_model
        self.class_num = config.class_num
        self.device = device

        self.tokenizer = BERTTokenizer(self.pretrained_model)
        self.model = BertModel.from_pretrained(self.pretrained_model)
        self.fc = nn.Linear(self.model.config.hidden_size, self.class_num)
        self.pos_ids = torch.arange(config.max_len).to(self.device)


    def forward(self, x, attn_mask):
        batch_size = x.size(0)
        pos_ids = self.pos_ids.expand(batch_size, -1)

        output = self.model(input_ids=x, attention_mask=attn_mask, position_ids=pos_ids)
        output = self.fc(output['pooler_output'])

        return output