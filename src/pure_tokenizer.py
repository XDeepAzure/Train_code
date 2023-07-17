from contextlib import contextmanager
from transformers import AutoTokenizer, NllbTokenizer
from typing import Any, List
import os
import json
from tqdm import tqdm
import numpy as np
import torch

class PureTokenizer(object):
    """
    抽取出new vocab 和new old vocab转换来使用
    """

    def __init__(self, tokenizer_path: str, vocab_path: str, convent_id: str, src_lang=None, tgt_lang=None) -> None:
        """
        """
        assert os.path.exists(tokenizer_path) and os.path.exists(vocab_path) and os.path.exists(convent_id), "vocab path and convent id 文件不存在"
        assert src_lang!=None and src_lang != tgt_lang, "src lang con't same as tgt lang"

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        pbar = tqdm(total=4)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        pbar.update()
        with open(vocab_path) as f:
            self.vocab = json.load(f)                                       # dict 形式 token2id
        self.vocab = {t: int(i) for t, i in self.vocab.items()}             # 将id转成int形式
        self.token2id = self.vocab
        self.id2token = {i:t for t,i in self.token2id.items()}

        with open(convent_id) as f:
            convent = json.load(f)

        self.new_token_id_2_old_token_id = convent["new_token_id_2_old_token_id"]
        self.old_token_id_2_new_token_id = convent["old_token_id_2_new_token_id"]
        pbar.update()
        self.old_token_id_2_new_token_id = {int(k):int(v) for k,v in self.old_token_id_2_new_token_id.items()}
        pbar.update()
        self.new_token_id_2_old_token_id = {int(k):int(v) for k,v in self.new_token_id_2_old_token_id.items()}
        pbar.update()

        # self.tokenizer.get
        for k, v in self.tokenizer.__dict__.items():
            if "__" not in k:
                self.__setattr__(k, v)
        # self.padding_side = self.tokenizer.padding_side
        # self.truncation_side = self.tokenizer.truncation_side
        self.pad = self.tokenizer.pad
        self.pad_token_id = self.tokenizer.pad_token_id

        pass

    def new_id_2_old_id(self, x: int|List[int]):
        """"""
        if isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
            x = x.tolist()
        if isinstance(x, tuple) or isinstance(x, list):
            return [self.new_id_2_old_id(y) for y in x]
        return self.new_token_id_2_old_token_id.get(x, self.tokenizer.unk_token_id)        # 如果取不到，那么就返回unk

    def old_id_2_new_id(self, x: int| List[int]):
        """"""
        if isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
            x = x.tolist()
        if isinstance(x, tuple) or isinstance(x, list):
            return [self.old_id_2_new_id(y) for y in x]
        return self.old_token_id_2_new_token_id.get(x, self.tokenizer.unk_token_id)        # 如果取不到，那么就返回unk
    
    def __call__(self, inputs, max_length=None, truncation=True, padding=False):

        inputs = self.tokenizer(inputs,
                                max_length=max_length,
                                truncation=truncation, padding=padding)
        # if isinstance(inputs["input_ids"], torch)                                         # ! 需要加上对于Tensor的判断
        inputs["input_ids"] = self.old_id_2_new_id(inputs["input_ids"])
        return inputs

    def distilled_input_ids(self, inputs, max_length=None, truncation=True, padding=False):
        return self.tokenizer(inputs, max_length=max_length, truncation=truncation, padding=padding)

    def convent_ids_to_tokens(self, ids):
        if isinstance(ids, tuple) or isinstance(ids, list):
            return [self.convent_ids_to_tokens(x) for x in ids]
        return self.id2token.get(ids, self.tokenizer.unk_token)
    
    def convent_tokens_to_ids(self, tokens):
        if isinstance(tokens, tuple) or isinstance(tokens, list):
            return [self.convent_tokens_to_ids(x) for x in tokens]
        return self.id2token.get(tokens, self.tokenizer.unk_token)

    def tokenize(self, x):
        return self.tokenizer.tokenize(x)
    
    def encode(self, x):
        enced = self.encode(x)
        enced = self.old_id_2_new_id(enced)
        return enced

    def batch_decode(self, x, skip_special_tokens=False):
        """可以作为batch使用
        """
        x = self.new_id_2_old_id(x)
        return self.tokenizer.batch_decode(x, skip_special_tokens=skip_special_tokens)

    def lang_code_to_id(self, lang):
        assert lang in (self.src_lang, self.tgt_lang), f"available lang is {self.src_lang, self.tgt_lang}"
        return self.tokenizer.lang_code_to_id(lang)

    def save_prtetrained(self, path):
        assert os.path.exists(path)
        self.tokenizer.save_pretrained(path)
        
        vocab_path = os.path.join(path, "id2token.json")
        convent_id = os.path.join(path, "convent_id_new_old.json")
        
        assert os.path.exists(vocab_path), f"path {vocab_path} has been exists"
        assert os.path.exists(convent_id), f"path {convent_id} has been exists"
        with open(vocab_path, "w") as f:
            json.dump(self.vocab, f, indent=4)
        with open(convent_id, "w") as f:
            json.dump({"new_token_id_2_old_token_id": self.new_token_id_2_old_token_id,
                       "old_token_id_2_new_token_id": self.old_token_id_2_new_token_id},
                       f, indent=4)
        pass

    def __str__(self) -> str:
        string = self.tokenizer.__str__()
        return string
        pass
    
    @contextmanager
    def as_target_tokenizer(self):
        src_lang = self.tokenizer.src_lang
        self.tokenizer.src_lang = self.tokenizer.tgt_lang
        #进入之前的代码
        yield
        #出来之前的代码
        self.tokenizer.src_lang = src_lang
        pass

    @staticmethod
    def from_pretrained(path, src_lang=None, tgt_lang=None):
        """

        """
        tokenizer_path = path
        vocab_path = os.path.join(path, "token2id.json")                    # token to id
        convent_id = os.path.join(path, "convent_id_new_old.json")
        tokenizer = PureTokenizer(tokenizer_path, vocab_path, convent_id, src_lang, tgt_lang)
        return tokenizer