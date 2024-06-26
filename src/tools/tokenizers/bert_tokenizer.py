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
        """
        Args:
            s (str): text.

        Returns:
            (list): tokenized text.
        """
        return self.tokenizer.tokenize(s)


    def encode(self, s):
        """
        Args:
            s (str): text.

        Returns:
            (list): tokens.
        """
        return self.tokenizer.encode(s, add_special_tokens=False)


    def decode(self, tok):
        """
        Args:
            tok (list): list of tokens.

        Returns:
            (str): decoded text.
        """
        return self.tokenizer.decode(tok)


    def convert_ids_to_tokens(self, tok_id):
        """
        Args:
            tok_id (int): token.

        Returns:
            (str): text token.
        """
        return self.tokenizer.convert_ids_to_tokens(tok_id)
    

    def convert_tokens_to_ids(self, tok_str):
        """
        Args:
            tok_str (str): text token.

        Returns:
            (int): token.
        """
        return self.tokenizer.convert_tokens_to_ids(tok_str)