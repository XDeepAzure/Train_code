import os
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from functools import partial
from datasets import load_from_disk, DatasetDict

from transformers import (M2M100ForConditionalGeneration,
                        #   Seq2SeqLMOutput,
                          AutoTokenizer)
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder, M2M100Model, shift_tokens_right
from src.train_args import parse_args
from src.utils import create_logger, setup_seed
from src import (Trainer, PureFFN, STEPS,
                distill_dec_step, distill_enc_step, teacher_forward, translate_step)
from eval import evaluate_both

from logging import getLogger

global logger
logger = getLogger()

def check_params(args):

    args.metrics = args.metrics.split(",")
    args.steps = [s for s in args.steps.split(",") if s in STEPS]

    args.saved_dir = os.path.join(args.saved_dir, args.name)
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

def main(args):
    check_params(args)
    setup_seed(args.seed)
    
    trainer = get_trainer(args)

    student = M2M100ForConditionalGeneration.from_pretrained(args.student_path)
    # if len(args.steps) > 1
    teacher = M2M100Model.from_pretrained(args.teacher_path)
    for p in teacher.parameters():
        p.requires_grad = False
    
    # ffn = PureFFN(args.input_size, args.hidden_size, args.output_size)
    ffn = PureFFN(1024, 1024, 1024)

    tokenizer = AutoTokenizer.from_pretrained(args.student_path)

    teacher, student, ffn = model_tocuda(teacher, student, ffn)

    trainer.model = {"student": student, "teacher": teacher, "ffn": ffn}
    trainer.tokenizer = tokenizer
    # 获取数据集
    data = load_from_disk(args.data_dir)
    data = DatasetDict({"train":data}) if not isinstance(data, DatasetDict) else data

    def train_steps_fn(model, x):
        loss = None
        return_metrics = dict()
        outputs = translate_step(model["student"], x)
        # outputs.pop("encoder_hidden_states")

        loss = outputs.loss
        return_metrics["translate_loss"] = loss.item()
        if len(args.steps) > 0:
            mask = x["labels"] != -100
            teacher_outputs = teacher_forward(model["teacher"], x, tokenizer.pad_token_id,
                                              decoder_start_token_id=model["student"].config.decoder_start_token_id)
            # teacher_outputs.pop("past_key_values")
        if STEPS[1] in args.steps or STEPS[3] in args.steps:
            distill_outputs = distill_enc_step(model["ffn"], teacher_outputs, outputs, x["attention_mask"], int(args.w_enc))
            loss += distill_outputs.pop("loss").to(loss.device)
            for k, v in distill_outputs.items():
                return_metrics[k] = v
        if STEPS[2] in args.steps or STEPS[3] in args.steps:
            distill_outputs = distill_dec_step(model["ffn"], teacher_outputs, outputs, mask, int(args.w_dec))
            loss += distill_outputs.pop("loss").to(loss.device)
            for k, v in distill_outputs.items():
                return_metrics[k] = v
        if STEPS[4] in args.steps:
            pass                            # 未完待续

        return loss, return_metrics

    trainer.train(train_steps_fn, data, evaluate_fn=evaluate_both, shuffle=args.shuffle)
    pass

if __name__ == "__main__":
    args = parse_args()
    main(args)