from datasets import Dataset, DatasetDict, load_dataset
from functools import partial
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
import numpy as np
import math
import os

from tqdm import tqdm

from datasets import load_from_disk, Dataset, DatasetDict

from logging import getLogger

FLORES_PATH = "/data/hyxu/cached_dir/flores"
CACHE_DIR = "/data/hyxu/cached_dir"

logger = getLogger()

class PureDataCollator(DataCollatorForSeq2Seq):

    def __call__(self, features, return_tensors=None):

        max_len = max([len(f["distilled_input_ids"]) for f in features])
        max_len = min(max_len, self.max_length)

        for f in features:
            input_ids = f["distilled_input_ids"]
            f['distilled_input_ids'] += [self.tokenizer.pad_token_id] * (max_len - len(input_ids)) 

        features = super().__call__(features, return_tensors)
        
        return features


def get_datasets_from_flores(src_lang_code, tgt_lang_code):
    """给src_lang_code 和tgt_lang_code 返回flores的valid 和test 数据集"""
    tokenized_datasets = load_dataset(FLORES_PATH, f"{src_lang_code}-{tgt_lang_code}", cache_dir=CACHE_DIR)
    # tokenized_datasets = load_dataset(FLORES_PATH, f"{src_lang_code}-{tgt_lang_code}")
    tokenized_datasets['test'] = tokenized_datasets.pop('devtest')
    tokenized_datasets['valid'] = tokenized_datasets.pop('dev')
    return tokenized_datasets
def preprocess_function(examples, src_lang, tgt_lang, tokenizer, max_input_length, max_target_length, is_pure=False):
    inputs = [ex for ex in examples[src_lang]]
    targets = [ex for ex in examples[tgt_lang]]
    ## ! 不要再tokenizer的时候padding
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Set up the tokenizer for targets 源语言与目标语言使用联合词典的
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def get_paras_from_file(*files):
    """_summary_
        files 的顺序必须是src， ref， pre， cor，后面的可以为空，但前面的必需有
    Returns:
        返回[[src],[ref], ...]
    """
    file_data = []
    for i, path in enumerate(files):
        with open(path, 'r') as f:
            f_data = f.readlines()
        f_data = [s.strip('\n').strip(" ") for s in f_data]
        file_data.append(f_data)

    # 过滤掉句子长度为0的句子
    trans_para = [item for item in zip(*file_data) if all([len(x)>0 for x in item])]
    logger.critical(f"过滤掉句子长度为0的句子，去掉了 {len(file_data[0]) - len(trans_para)} 个句子")
    return trans_para

def get_translate_paras_from_file(src_file, tgt_file):
    trans_paras = get_paras_from_file(src_file, tgt_file)   #这里tgt_file作为ref传进去的
    assert len(trans_paras[0]) == 2, "平行语句不对应"
    return trans_paras


def get_tokenized_datasets(tokenizer, trans_para, src_lang, tgt_lang, max_input_length, max_target_length, batch_size=None):
    """
    注意 着里的trans_para 只能是有两个元素的，分别作为源语言和目标语言, 也可以是datasetdict
    只进行tokenized不做split trans_para 可以是list也可以是DatasetDict
    """
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    batch_tokenize_fn = partial(preprocess_function,
                                tokenizer=tokenizer,
                                src_lang=src_lang,
                                tgt_lang=tgt_lang,
                                max_input_length=max_input_length,
                                max_target_length=max_target_length,
                                )
    if not isinstance(trans_para, DatasetDict):
        trans_para = {
            src_lang: [src for src, _ in trans_para],
            tgt_lang: [tgt for _, tgt in trans_para]
        }
        raw_datasets = Dataset.from_dict(trans_para)
        raw_datasets = DatasetDict({'train': raw_datasets})
    else:
        raw_datasets = trans_para
    remove_names = raw_datasets['train'].column_names if "train" in raw_datasets else raw_datasets['test'].column_names

    tokenized_datasets = raw_datasets.map(batch_tokenize_fn, batched=True, batch_size=batch_size,
                                          remove_columns=remove_names)
    return tokenized_datasets

def get_dataloader(data: Dataset,  data_collator=None, tokenizer=None, batch_size=32, max_length=256, padding=True, shuffle=False):
    if data_collator == None:
        assert tokenizer != None, "data_collator 与tokenizer不能同时为None"
        data_collator = DataCollatorForSeq2Seq(tokenizer, padding=padding, max_length=max_length)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator)
    return dataloader

# def get_dataloader(tokenizer, src_lang, tgt_lang, batch_size=32):
        
#     raw_dataset = get_datasets_from_flores(tokenizer.src_lang, tokenizer.tgt_lang)
#     raw_dataset.pop("valid")
#     tokenized_datasets = get_tokenized_datasets(tokenizer, raw_dataset,
#                                                 f"sentence_{tokenizer.src_lang}",
#                                                 f"sentence_{tokenizer.tgt_lang}",
#                                                 max_input_length=256,
#                                                 max_target_length=256, batch_size=batch_size)
#     data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, max_length=256)
#     dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size,
#                             collate_fn=data_collator, shuffles=Fale)
#     return dataloader
# dataloader = get_dataloader()

def get_data_from_flore(langs, split="test"):
    assert len(langs) >= 1
    data = dict()
    split = "devtest" if split=="test" else "dev"
    for lang in langs:
        data[lang] = load_dataset(FLORES_PATH, lang, cache_dir=CACHE_DIR)[split][f"sentence"]
    return data

def load_translate_datasets(data_dir, src_lang, tgt_lang, src_file, tgt_file, tokenizer, max_length, batch_size, bi=False):
    """
    return DatasetDict {"train": DataSet, "dev": {"src_lang":[], "tgt_lang":[]}}
    """
    if bi:                                                                     # 是不是用的双向翻译模型
        if not os.path.exists(os.path.join(data_dir, f"both_{src_lang}_{tgt_lang}")):
            trans_para = get_paras_from_file(os.path.join(data_dir, src_file[0]), os.path.join(data_dir, tgt_file[0]))
            data = get_tokenized_datasets(tokenizer=tokenizer, trans_para=trans_para, src_lang=src_lang, tgt_lang=tgt_lang,
                                          max_input_length=max_length, max_target_length=max_length, batch_size=batch_size)
            data1 = data["train"].to_dict()
            trans_para = get_paras_from_file(os.path.join(data_dir, tgt_file[0]), os.path.join(data_dir, src_file[0]))
            data = get_tokenized_datasets(tokenizer=tokenizer, trans_para=trans_para, src_lang=tgt_lang, tgt_lang=src_lang,
                                          max_input_length=max_length, max_target_length=max_length, batch_size=batch_size)
            data = {k: v+data1[k] for k, v in data["train"].to_dict().items()}
            data_dict = DatasetDict({"train": Dataset.from_dict(data)})
            data_dict.save_to_disk(os.path.join(data_dir, f"both_{src_lang}_{tgt_lang}"))
        else:
            data_dict = load_from_disk(os.path.join(data_dir, f"both_{src_lang}_{tgt_lang}"))
    else:
        if not os.path.exists(os.path.join(data_dir, f"{src_lang}-{tgt_lang}")):
            trans_para = get_paras_from_file(os.path.join(data_dir, src_file[0]), os.path.join(data_dir, tgt_file[0]))
            data_dict = get_tokenized_datasets(tokenizer=tokenizer, trans_para=trans_para, src_lang=src_lang, tgt_lang=tgt_lang,
                                          max_input_length=max_length, max_target_length=max_length, batch_size=batch_size)
            data_dict.save_to_disk(os.path.join(data_dir, f"{src_lang}-{tgt_lang}"))
        else:
            data_dict = load_from_disk(os.path.join(data_dir, f"{src_lang}-{tgt_lang}"))
    test = Dataset.load_from_disk("/data/hyxu/lowMT_compute/data/public_data/dev_set")
    data_dict["dev"] = {f"{src_lang}-{tgt_lang}": test}
    if bi:
        data_dict["dev"][f"{tgt_lang}-{src_lang}"] = test
    return data_dict

def load_denoising_datasets(data_dir, denoising_file, lang, tokenizer, max_length, batch_size):
    dataset_path = os.path.join(data_dir, f"{lang}-denosing")
    if os.path.exists(dataset_path):
        data = load_from_disk(dataset_path)
    else:
        def _filter(s):
            return s.strip(" ").strip("\n")
        with open(os.path.join(data_dir, denoising_file), "r") as f:
            data = f.readlines()
        data = [_filter(s) for s in data if len(s) > 5]
        data_intputs, labels, mask = [], [], []
        tokenizer.src_lang = lang
        for s in tqdm(data):
            inputs = tokenizer(s, max_length=max_length, truncation=True)
            labels.append(inputs["input_ids"])
            nosie_inputs = add_span_mask_noise(inputs["input_ids"])
            data_intputs.append(nosie_inputs)
            mask.append([1 for _ in nosie_inputs])
        data = Dataset.from_dict({"input_ids": data_intputs, "attention_mask": mask, "labels": labels})
        data.save_to_disk(dataset_path)
        
    return data


def permute_sentences(self, source, p=1.0):
    """文档中的句子随机扰乱"""
    full_stops = source == self.full_stop_index
    # Pretend it ends with a full stop so last span is a sentence
    full_stops[-2] = 1
    # Tokens that are full stops, where the previous token is not
    sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero(as_tuple=False) + 2
    result = source.clone()
    num_sentences = sentence_ends.size(0)
    num_to_permute = math.ceil((num_sentences * 2 * p) / 2.0)
    substitutions = torch.randperm(num_sentences)[:num_to_permute]
    ordering = torch.arange(0, num_sentences)
    ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]
    # Ignore <bos> at start
    index = 1
    for i in ordering:
        sentence = source[(sentence_ends[i - 1] if i > 0 else 1) : sentence_ends[i]]
        result[index : index + sentence.size(0)] = sentence
        index += sentence.size(0)
    return result

def word_starts(source):
    """判断是不是句子的开始"""
    is_word_start = torch.ones(source.size())
    is_word_start[0] = 0
    is_word_start[-1] = 0
    return is_word_start

def add_permuted_noise(self, tokens, p):
    num_words = len(tokens)
    num_to_permute = math.ceil(((num_words * 2) * p) / 2.0)
    substitutions = torch.randperm(num_words - 2)[:num_to_permute] + 1
    tokens[substitutions] = tokens[substitutions[torch.randperm(num_to_permute)]]
    return tokens
def add_rolling_noise(self, tokens):
    offset = np.random.randint(1, max(1, tokens.size(-1) - 1) + 1)
    tokens = torch.cat(
        (tokens[0:1], tokens[offset:-1], tokens[1:offset], tokens[-1:]),
        dim=0,
    )
    return tokens
def add_insertion_noise(self, tokens, p):
    if p == 0.0:
        return tokens
    num_tokens = len(tokens)
    n = int(math.ceil(num_tokens * p))
    noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
    noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
    noise_mask[noise_indices] = 1
    result = torch.LongTensor(n + len(tokens)).fill_(-1)
    num_random = int(math.ceil(n * self.random_ratio))
    result[noise_indices[num_random:]] = self.mask_idx
    result[noise_indices[:num_random]] = torch.randint(
        low=1, high=len(self.vocab), size=(num_random,)
    )
    result[~noise_mask] = tokens
    assert (result >= 0).all()
    return result

def create_sentinel_ids(mask_indices, vocab_size):
    start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices

    sentinel_ids = np.where(
        start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
    )
    # making sure all sentinel tokens are unique over the example
    sentinel_ids = np.where(sentinel_ids != 0, vocab_size - sentinel_ids, 0)
    sentinel_ids -= mask_indices - start_indices
    return sentinel_ids

def filter_input_ids(input_ids, sentinel_ids):
    """
    Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
    This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
    """
    input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)

    # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
    # masked tokens coming after sentinel tokens and should be removed
    return input_ids_full[input_ids_full >= 0]

def random_spans_noise_mask(noise_density, mean_noise_span_length, length):

        """
        This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """
            Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of subsegments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans
        )

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

def add_span_mask_noise(item):
    noise_mask = random_spans_noise_mask(len(item))

    source_sentinel_ids = create_sentinel_ids(noise_mask.astype(np.int8))
    source = filter_input_ids(item, source_sentinel_ids)
    return source

    # target_sentinel_ids = create_sentinel_ids(
    #     (~noise_mask).astype(np.int8)
    # )
    # target = filter_input_ids(item, target_sentinel_ids)
