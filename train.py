import os
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from functools import partial
from datasets import load_from_disk, DatasetDict, Dataset

from transformers import (AutoModelForSeq2SeqLM,
                        #   Seq2SeqLMOutput,
                          AutoTokenizer)
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder, M2M100Model, shift_tokens_right
from src.train_args import parse_args
from src.utils import create_logger, setup_seed
from src import (Trainer, STEPS, translate_step, get_tokenized_datasets, get_paras_from_file)
from eval import evaluate_both, evaluate_fn

from logging import getLogger

global logger
logger = getLogger()

def check_params(args):

    args.metrics = args.metrics.split(",")
    args.steps = [s for s in args.steps.split(",") if s in STEPS]

    args.saved_dir = os.path.join(args.saved_dir, args.name)
    args.src_file = args.src_file.split(",")
    args.tgt_file = args.tgt_file.split(",")
    if not os.path.exists(args.saved_dir):
        os.mkdir(args.saved_dir)
    global logger
    log_name = "train.log"
    logger = create_logger(os.path.join(args.saved_dir, log_name))
    logger.info(f"process id is {os.getpid()}")
    logger.info(args.des)
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(args)).items())))
    pass

def get_trainer(args):
    """
    # return Trainer(accumulation=getattr(args, "accumulation", 1),
    #                max_norm=getattr(args, "max_norm", 2),
    #                max_length=getattr(args, "max_length", 128),
    #                num_beams=getattr(args, "num_beams", 4),
    #                batch_size=getattr(args, "batch_size", 32),
    #                seed=getattr(args, "seed"))
    """
    trainer = Trainer()
    for k in dir(args):
        if not callable(getattr(args, k, None)):
            if not k.startswith("_") and hasattr(trainer, k):
                setattr(trainer, k, getattr(args, k))
    return trainer

def model_tocuda(teacher_model, student_model, ffn_model):
    print("把模型放到cuda上")
    device1 = torch.device("cuda:0")
    device2 = torch.device("cuda:1")

    teacher_model = teacher_model.to(device1)
    ffn_model = ffn_model.to(device1)
    student_model = student_model.to(device2)                                       # 学生放在第二张卡上
    return teacher_model, student_model, ffn_model

def get_DatasetDict(data_dir, src_lang, tgt_lang, src_file, tgt_file, tokenizer, max_length, batch_size, bi=False) -> DatasetDict:
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
    
    test = Dataset.load_from_disk("/data/hyxu/codes/lowMT_compute/data/public_data/dev_set")

    data_dict["dev"] = {f"{src_lang}-{tgt_lang}": test}
    if bi:
        data_dict["dev"][f"{tgt_lang}-{src_lang}"] = test    
    return data_dict

def evaluate_(model, tokenizer, data=None, batch_size=32, num_beams=4,
                  max_length=128, metrics=["chrf"], split="test", output_dir=None, save_text=False):
    avg_scores = []
    return_metrics = dict()
    logger.info("==============evaluate begin===============")
    for k, v in data.items():
        s, t = k.split("-")
        outputs = evaluate_fn(model["model"], tokenizer, s, t, {s:v[s], t:v[t]}, batch_size=batch_size, num_beams=num_beams, max_length=max_length,
                    metrics=metrics)
        outputs = outputs[0]
        for m, s in outputs.items():
            return_metrics[f"{k}/{m}"] = s
        avg_scores.append(outputs[metrics[0]])
    return_metrics[f"avg_{metrics[0]}"] = sum(avg_scores) / len(avg_scores)
    logger.info(return_metrics)
    return return_metrics
        
        

def main(args):
    check_params(args)
    setup_seed(args.seed)
    
    trainer = get_trainer(args)

    student = AutoModelForSeq2SeqLM.from_pretrained(args.student_path)

    tokenizer = AutoTokenizer.from_pretrained(args.student_path)

    student = student.cuda()

    trainer.model = {"model": student}
    trainer.tokenizer = tokenizer
    # 获取数据集
    data = get_DatasetDict(data_dir=args.data_dir, src_lang=args.src_lang, tgt_lang=args.tgt_lang,
                           src_file=args.src_file, tgt_file=args.tgt_file,
                           tokenizer=tokenizer, max_length=args.max_length, batch_size=args.batch_size, bi=args.bi)

    def train_steps_fn(model, x):
        loss = None
        return_metrics = dict()
        outputs = translate_step(model["model"], x)
        loss = outputs.loss
        return_metrics["translate_loss"] = outputs.loss.item()
        if trainer.label_smoother:
            outputs = trainer.label_smooth_step(outputs=outputs, labels=x["labels"], shift_labels=False)
            loss = outputs.loss
            return_metrics["label_smooth_loss"] = loss.item()
        

        return loss, return_metrics

    trainer.train(train_steps_fn, data, evaluate_fn=evaluate_, shuffle=args.shuffle)
    pass

if __name__ == "__main__":
    args = parse_args()
    main(args)