import os
from tqdm import tqdm
import json
import torch

# 自动混合精度

from transformers import (M2M100ForConditionalGeneration,
                          MBartForConditionalGeneration,
                          AutoModelForSeq2SeqLM,
                          AutoTokenizer,
                          DataCollatorForSeq2Seq)

from datasets import load_from_disk, DatasetDict
from torch.utils.data import DataLoader

from torch.cuda.amp import autocast as autocast

import fire

from src import get_data_from_flore
from src.utils import langs, create_logger, setup_seed
from logging import getLogger

global logger
logger = getLogger()

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def get_compute_metric_fn(metrics):
    compute_fn = dict()
    if "chrf" in metrics:
        from sacrebleu import CHRF
        chrf = CHRF(word_order=2)
        compute_fn["chrf"] = chrf.corpus_score
    if "bleu" in metrics:
        from sacrebleu import BLEU
        bleu = BLEU(tokenize="char")
        compute_fn["bleu"] = bleu.corpus_score
    return compute_fn

def generate(model, tokenizer, src_lang, tgt_lang, src_data, batch_size=32, num_beams=4, max_length=128):
    num_batch = len(src_data) // batch_size if len(src_data) % batch_size == 0 else len(src_data) // batch_size +1
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(num_batch)):
            x = tokenizer(src_data[i*batch_size:(i+1)*batch_size], padding=True, max_length=max_length, return_tensors="pt",truncation=True)
            for k, v in x.items():
                x[k] = v.to(model.device)
            with autocast():
                if isinstance(model, MBartForConditionalGeneration):
                    outputs = model.generate(input_ids=x["input_ids"],
                                     attention_mask=x["attention_mask"],
                                     num_beams=num_beams,
                                     decoder_start_token_id=tokenizer.lang_code_to_id[tgt_lang])            #mbart
                elif isinstance(model, M2M100ForConditionalGeneration):
                    outputs = model.generate(input_ids=x["input_ids"],
                                     attention_mask=x["attention_mask"],
                                     num_beams=num_beams,
                                     forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])             #nllb
            predictions += tokenizer.batch_decode(outputs.tolist(), skip_special_tokens=True)
    return predictions

def evaluate_fn(model, tokenizer, src_lang, tgt_lang, data, batch_size=32, num_beams=4, max_length=128, metrics=["chrf"]):
    """
    data = {src_lang: [], tgt_lang: []}
    return {"bleu": 20, "chrf":30}, predictions, references
    """
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    model.eval()
    if model.device == torch.device("cpu"):
        model = model.cuda()
        
    references = data[tgt_lang]
    predictions = generate(model, tokenizer, src_lang, tgt_lang, data[src_lang],
                           batch_size=batch_size, num_beams=num_beams, max_length=max_length)
    return_metrics = dict()
    for k, v in get_compute_metric_fn(metrics).items():
        return_metrics[k] = v(hypotheses=predictions, references=[references]).score

    return return_metrics, predictions, references

def get_dataloader_from_dataset(data_path, batch_size, max_input_length=128, shuffle=True):
    valid = load_from_disk(f"{data_path}/valid")
    test = load_from_disk(f"{data_path}/test")
    train = load_from_disk(f"{data_path}/train")
    tokenized_datasets = DatasetDict({"train": train, "valid": valid, "test": test})
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, max_length=max_input_length)

    def _get_loader(data, batch_size):
        if isinstance(data, dict):
            for k, v in data.items():
                data[k] = _get_loader(v, batch_size)
            return data
        else:
            return DataLoader(data, batch_size=batch_size,
                                collate_fn=data_collator, shuffle=shuffle)
    tokenized_datasets["train"] = _get_loader(tokenized_datasets["train"], batch_size)
    tokenized_datasets["valid"] = _get_loader(tokenized_datasets["valid"], batch_size*2)
    tokenized_datasets["test"] = _get_loader(tokenized_datasets["test"], batch_size*2)
    return tokenized_datasets["train"], tokenized_datasets["valid"], tokenized_datasets["test"]

def evaluate(model, data_loader, tgt_lang=None):
    model.eval()
    if tgt_lang == None:
        tgt_lang = tokenizer.tgt_lang
    with torch.no_grad():
        predictions = []
        references = []
        for x in tqdm(data_loader, total=len(data_loader)):
            for k, v in x.items():
                x[k] = v.to(model.device)
            outputs = model.generate(input_ids=x["input_ids"],
                                     attention_mask=x["attention_mask"],
                                     num_beams=num_beams,
                                     forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
            predictions += tokenizer.batch_decode(outputs.tolist(), skip_special_tokens=True)
            x["labels"][x["labels"]==-100] = tokenizer.pad_token_id
            references += tokenizer.batch_decode(x["labels"].tolist(), skip_special_tokens=True)
    from sacrebleu import CHRF, BLEU
    chrf = CHRF(word_order=2)
    bleu = BLEU()
    # score = [chrf.sentence_score(hypothesis=p, references=[r]).score for p, r in zip(predictions, references)]
    chrf_score = chrf.corpus_score(hypotheses=predictions, references=[references])
    bleu_score = bleu.corpus_score(hypotheses=predictions, references=[references])
    # import pdb;pdb.set_trace()
    return {"chrf": chrf_score.score, "bleu": bleu_score.score}, predictions, references


# def evaluate_fn(student_model, split="valid", update_step=0, scores=dict()):
#     print("--------- evaluating -----")
#     dataloader = valid_loader
#     if split == "test":
#         dataloader = test_loader
#     if isinstance(dataloader, dict):
#         avg_chrf, avg_bleu = [], []
#         for k, v in tqdm(dataloader.items()):
#             metrics = evaluate(student_model, v, k.split("-")[-1])
#             metric = metrics[0]
#             scores[k] = metric["chrf"]
#             avg_chrf.append(metric["chrf"])
#             avg_bleu.append(metric["bleu"])
#         metrics = ({"chrf": sum(avg_chrf)/len(avg_chrf), "bleu": sum(avg_bleu)/ len(avg_bleu)}, )
#         # for m_k, m_v in metrics.items():
#         #     writer.add_scalar(f"{split}/{k}/avg", m_v, update_step)
#     return metrics


def evaluate_to_en(model, tokenizer, data=None, batch_size=32, num_beams=4,
                   max_length=128, metrics=["chrf"], split="test", output_dir=None, save_text=False):
    """
    data = {lang1: [], lang2: [], ... }
    """
    en_lang = "eng_Latn"
    if data is None:
        data = get_data_from_flore(langs, split=split)
    metric_score = dict()
    for lang in tqdm(langs):
        if lang == en_lang:
            continue
        outputs = evaluate_fn(model, tokenizer, lang, en_lang, {lang: data[lang], en_lang: data[en_lang]},
                    batch_size, num_beams, max_length, metrics)
        metric_score[lang] = outputs[0]
    if output_dir != None:
        with open(f"{output_dir}/metric_to_en.json", "w") as f:
            json.dump(metric_score, f, indent=4)
    if save_text:
        pass
    metric_score = {f"{k}/chrf": v["chrf"] for k, v in metric_score.items()}
    logger.info(f"on {split} set to eng metric is {metric_score}")
    logger.info(f"avg is {sum(metric_score.values()) / len(metric_score.values())}")
    return metric_score

def evaluate_from_en(model, tokenizer, data=None, batch_size=32, num_beams=4,
                     max_length=128, metrics=["chrf"], split="test", output_dir=None, save_text=False):
    en_lang = "eng_Latn"
    metric_score = dict()
    if data is None:
        data = get_data_from_flore(langs, split=split)
    for lang in tqdm(langs):
        if lang == en_lang:
            continue
        outputs = evaluate_fn(model, tokenizer, en_lang, lang, {en_lang: data[en_lang], lang: data[lang]},
                    batch_size, num_beams, max_length, metrics)
        metric_score[lang] = outputs[0]
    if output_dir != None:
        with open(f"{output_dir}/metric_from_en.json", "w") as f:
            json.dump(metric_score, f, indent=4)
    if save_text:
        pass
    metric_score = {f"{k}/chrf": v["chrf"] for k, v in metric_score.items()}
    logger.info(f"on {split} set from eng metric is {metric_score}")
    logger.info(f"avg is {sum(metric_score.values()) / len(metric_score.values())}")
    return metric_score

def evaluate_both(model, tokenizer, data=None, batch_size=32, num_beams=4,
                  max_length=128, metrics=["chrf"], split="test", output_dir=None, save_text=False):
    score = dict()
    metrics_socre = evaluate_to_en(model, tokenizer, batch_size=batch_size,
                                       num_beams=num_beams, max_length=max_length, metrics=metrics,
                                       split=split, output_dir=output_dir, save_text=save_text)
    for k, v in metrics_socre.items():
        score[f"to_{k}"] = v
    metrics_socre = evaluate_from_en(model, tokenizer, batch_size=batch_size,
                                       num_beams=num_beams, max_length=max_length,metrics=metrics,
                                       split=split, output_dir=output_dir, save_text=save_text)
    for k, v in metrics_socre.items():
        score[f"from_{k}"] = v
    return score


def load_dataset(src_lang, tgt_lang, src_file, tgt_file, data_dir, test_dataset):
    """return {"src_lang": [], "tgt_lang": []}"""
    if test_dataset != "":
        data = load_from_disk(os.path.join(data_dir, test_dataset))
    else:
        def _read_file(path):
            with open(os.path.join(data_dir, path)) as f:
                data = f.readlines()
            return [s.rstrip("\n") for s in data]
        data = {src_lang: _read_file(src_file), tgt_lang: _read_file(tgt_file)}
    return data


def main(model_path: str, src_lang:str, tgt_lang:str, src_file:str=None, tgt_file:str=None,
         data_dir: str = None, seed=235, output_dir: str = None, metrics: str = "bleu,",
         num_beams=5, max_length:int = 128, batch_size: int = 32, test_dataset: str = "", split="test", save_text=False):
    # model_path = "/data/hyxu/cached_dir/nllb-200-distilled-600M"

    global logger
    output_dir = model_path if output_dir is None else output_dir
    metrics = [m for m in metrics if len(m) >= 1]
    setup_seed(seed)

    log_name = f"{output_dir}/eval.log"
    logger = create_logger(log_name)
    logger.info(f"procss is {os.getpid()}")
    logger.info(f"num_beams : {num_beams} max_length: {max_length}")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang=src_lang, tgt_lang=tgt_lang)
    
    data = load_dataset(src_lang, tgt_lang, src_file, tgt_file, data_dir, test_dataset)

    metrics_socre, predictions, references = evaluate_fn(model=model, tokenizer=tokenizer,
                                src_lang=src_lang, tgt_lang=tgt_lang, data=data,
                                batch_size=batch_size, num_beams=num_beams,
                                max_length=max_length, metrics=metrics)
    logger.info(metrics_socre)
    if save_text:
        with open(f"{output_dir}/predictions.txt", "w") as f:
            f.writelines([s+"\n" for s in predictions])
        with open(f"{output_dir}/references.txt", "w") as f:
            f.writelines([s+"\n" for s in references])

    return metrics_socre




if __name__ == "__main__":
    fire.Fire(main)