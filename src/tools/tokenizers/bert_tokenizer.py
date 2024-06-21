from transformers import BertTokenizer



class BERTTokenizer:
    def __init__(self, path):
        self.tokenizer = BertTokenizer.from_pretrained(path)

        self.bos_token, self.bos_token_id = self.tokenizer.bos_token, self.tokenizer.bos_token_id
        self.eos_token, self.eos_token_id = self.tokenizer.eos_token, self.tokenizer.eos_token_id
        self.pad_token, self.pad_token_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id
        self.cls_token, self.cls_token_id = self.tokenizer.cls_token, self.tokenizer.cls_token_id
        self.sep_token, self.sep_token_id = self.tokenizer.sep_token, self.tokenizer.sep_token_id
        self.unk_token, self.unk_token_id = self.tokenizer.unk_token, self.tokenizer.unk_token_id
        self.msk_token, self.msk_token_id = self.tokenizer.mask_token, self.tokenizer.mask_token_id

        self.vocab_size = len(self.tokenizer)


    def tokenize(self, s):
        return self.tokenizer.tokenize(s)


    def encode(self, s):
        # for eliminate sos and eos tokens that are automatically added to a sentence
        return self.tokenizer.encode(s)[1:-1]


    def decode(self, tok):
        try:
            tok = tok[:tok.index(self.pad_token_id)]
        except:
            pass
        return self.tokenizer.decode(tok)