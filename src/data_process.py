from datasets import Dataset, DatasetDict, load_dataset
from functools import partial
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

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