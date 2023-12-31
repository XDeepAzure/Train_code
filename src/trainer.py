import os
from logging import getLogger
import shutil

import torch

from tqdm import tqdm
from transformers import (AutoTokenizer, DataCollatorForSeq2Seq)

from torch.utils.data import DataLoader

# 自动混合精度
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

from src.model import PureFFN
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder, M2M100Model, shift_tokens_right

from torch.utils.tensorboard import SummaryWriter

logger = getLogger()




OPTIMIZER = ("AdamW", "SGD")
STRATEGY = ("steps", "epoch")


def avg(x):
    assert isinstance(x[0], int)
    return sum(x) / len(x)

def to_same_device(x, y):
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = to_same_device(v, y)
    elif x.device != y.device:
        x = x.to(y.device)
    return x

def evaluate_fn(model, tokenizer, data_loader, tgt_lang, num_beams=4, return_text=False):
    from sacrebleu import CHRF, BLEU
    chrf = CHRF(word_order=2)
    bleu = BLEU()

    predictions, references = [], []
    model.eval()
    with torch.no_grad():
        for x in tqdm(data_loader, total=len(data_loader)):
            for k, v in x.items():
                x[k] = v.to(model.device)
            outputs = model.generate(**x, num_beams=num_beams,
                                     forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
            predictions += tokenizer.batch_decode(outputs, skip_special_tokens=True)

        x["labels"][x["labels"]==-100] = tokenizer.pad_token_id
        references += tokenizer.batch_decode(x["labels"], skip_special_tokens=True)

    chrf_score = chrf.corpus_score(hypotheses=predictions, references=[references])
    bleu_score = bleu.corpus_score(hypotheses=predictions, references=[references])
    # import pdb;pdb.set_trace()
    if return_text:
        return {"chrf": chrf_score.score, "bleu": bleu_score.score}, predictions, references
    else:
        return {"chrf": chrf_score.score, "bleu": bleu_score.score}

def translate_step(model, x):
    """ 翻译任务的步骤，包含 encoder_last_hidden_state,  decoder_hidden_states(12层)"""
    for k, v in x.items():
        x[k] = v.to(model.device)
    return model(**x, output_hidden_states=True)

def teacher_forward(model, x, pad_token_id, decoder_start_token_id):
    """返回teacher的forward结果 包含encoder_last_hidden_state, last_hidden_state"""
    x["decoder_input_ids"] = shift_tokens_right(x["labels"], pad_token_id, decoder_start_token_id)
    x.pop("labels")
    for k, v in x.items():
        x[k] = v.to(model.device)
    return model(input_ids=x["input_ids"], attention_mask=x["attention_mask"],
                 decoder_input_ids=x["decoder_input_ids"], output_hidden_states=False)

def distill_enc_step(ffn_model, teacher_outputs, student_outputs, attention_mask, w):
    """所有的step都以返回的loss为要backward的loss，后面的就是要记录的metric"""
    if isinstance(teacher_outputs, dict):
        student_encoder_hidden_states = student_outputs.encoder_last_hidden_state
        teacher_encoder_hidden_states = teacher_outputs.encoder_last_hidden_state
    else:
        student_encoder_hidden_states = student_outputs
        teacher_encoder_hidden_states = teacher_outputs
    enc_loss = ffn_model(x=student_encoder_hidden_states,
                         y=teacher_encoder_hidden_states, y_mask=attention_mask)
    # import pdb;pdb.set_trace()
    return {"loss": enc_loss * w, "enc_loss": enc_loss.item()}

def distill_dec_step(ffn_model, teacher_outputs, student_outputs, attention_mask, w):
    """
    attention_mask 是label的mask
    """
    if isinstance(teacher_outputs, dict):
        teacher_decoder_hidden_states = teacher_outputs.last_hidden_state
        student_deocder_hidden_states = student_outputs.decoder_hidden_states[-1]               # seq2seqoutput
    else:
        teacher_decoder_hidden_states = teacher_outputs
        student_deocder_hidden_states = student_outputs
    dec_loss = ffn_model(x=student_deocder_hidden_states,
                         y=teacher_decoder_hidden_states,y_mask=attention_mask)
    return {"loss": dec_loss * w, "dec_loss": dec_loss.item()}

def dec_noise_step(model, teacher_outputs, student_outputs, attention_mask, w):
    """对decoder的输出的特征加噪音，然后在过lm_head 做交叉熵"""
    if isinstance(teacher_outputs, dict):
        teacher_decoder_hidden_states = teacher_outputs.last_hidden_state
        student_deocder_hidden_states = student_outputs.decoder_hidden_states[-1]               # seq2seqoutput
    else:
        teacher_decoder_hidden_states = teacher_outputs
        student_deocder_hidden_states = student_outputs
    teacher_decoder_hidden_states = teacher_decoder_hidden_states.to(student_deocder_hidden_states.device)
    attention_mask = attention_mask.to(student_deocder_hidden_states.device)

    teacher_hidden = teacher_decoder_hidden_states[attention_mask]
    student_hidden = student_deocder_hidden_states[attention_mask]
    # 均值
    x = (student_hidden - teacher_hidden).sum(0) / student_hidden.shape(-1)      # (hidden_size)

    # 方差




class Trainer(object):

    def __init__(self, model=None, tokenizer=None, accumulation=1, max_norm=2, max_length=128, num_beams=4,
                 batch_size=32, seed=10, saved_dir="./", shuffle=True, datasets=None,
                 train_strategy=STRATEGY[0], eval_strategy=STRATEGY[0], save_step=1, eval_step=1, log_step=100,
                 num_epoch=3, max_step=10000, metrics=["chrf"], amp=True, optimizer=OPTIMIZER[0], lr=2e-5,
                 weight_decay=0.01, save_num=3) -> None:

        # self.optimizer                                    # 默认optimizer是adamw
        # self.check_params(args)
        
        self.model = model
        self.tokenizer = tokenizer

        self.saved_dir = saved_dir
        self.seed = seed
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.max_length = max_length

        assert train_strategy in STRATEGY and eval_strategy in STRATEGY, ""
        self.train_strategy = train_strategy                # 二者策略没用任何影响
        self.eval_strategy = eval_strategy

        self.log_step = log_step
        assert save_step % eval_step == 0, ""               # 此条件必须成立
        self.eval_step = eval_step
        self.save_step = save_step

        self.num_epoch = num_epoch
        self.max_step = max_step

        self.metrics = metrics

        self.shuffle = shuffle
        self.datasets = datasets                              # 数据集作为字典存储
        # 梯度相关的参数
        self.accumulation = accumulation
        self.max_norm = max_norm

        self.model = dict()                                 # 字典存储模型
        self.epoch_size = -1

        ## TODO 优化器参数 所设置的均为默认值
        self.amp = amp
        self.weight_decay = weight_decay
        self.lr = lr
        assert optimizer in OPTIMIZER
        self.optimizer = optimizer
        
        self.parameters = None
        self.best_model = []
        self.save_num = save_num
        self.update_step = 0

        pass

    def check_params(self, args):
        """设置一些默认值"""

        # if not hasattr(args, "input_size"):
        #     args.input_size = 1024                          # nllb-600M
        # if not hasattr(args, "hidden_size"):
        #     args.input_size = 1280                          # 1024 + 256
        # if not hasattr(args, "input_size"):
        #     args.input_size = 2048                          # nllb-3.3B
        
        
        if not hasattr(args, "save_dir"):
            args.save_dir = os.path.abspath()
        if not hasattr(args, "seed"):
            args.seed = 10
        if not hasattr(args, "batch_size"):
            args.batch_size = 16
        if not hasattr(args, "eval_batch_size"):
            args.eval_batch_size = args.batch_size * 2
        

        ## TODO step 相关
        if not hasattr(args, "logging_steps"):
            args.logging_steps = 100
        if not hasattr(args, "save_steps"):
            args.save_steps = 2000
        if not hasattr(args, "eval_steps"):
            args.eval_steps = 2000
        if not hasattr(args, "max_steps"):
            args.max_steps = None                                   # 那么这就是以epoch为一循环
        
        assert args.save_steps % args.eval_steps == 0, "save step 必须是eval step 的整数倍"

        
        ## TODO 优化器参数 所设置的均为默认值
        if not hasattr(args, "optimizer"):
            args.optimizer = OPTIMIZER[0]
        assert args.optimizer in OPTIMIZER, f"{args.optimizer}优化器目前不支持"
        if not hasattr(args, "lr"):
            args.lr = 4e-6
        if not hasattr(args, "weight_decay"):
            args.weight_decay = 0.01
        if not hasattr(args, "amp"):
            args.amp = True
        
        

        ## TODO 模型相关的
        if not hasattr(args, "num_beams"):
            args.num_beams = 1
        if not hasattr(args, "max_input_length"):
            args.max_input_length = 256
        if not hasattr(args, "max_target_length"):
            args.max_target_length = 256
        

        self.args = args

    def init_optimizer(self):
        if self.parameters == None or len(self.parameters) < 1:
            self.parameters = []
            for v in self.model.values():
                self.parameters += [p for p in v.parameters() if p.requires_grad == True]
        ## ! 暂未完成
        if self.optimizer == OPTIMIZER[0]:
            self.optimizer = torch.optim.AdamW(params=self.parameters,
                                               lr=self.lr,
                                               weight_decay=self.weight_decay)
        else:
            assert 1==2, "优化器暂未完成"
        logger.info(self.optimizer)
        return self.optimizer
    
    def _create_dataloader(self, dataset, batch_size, shuffle=False):
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, padding=True, 
                                               max_length=self.max_length)

        return iter(DataLoader(dataset, batch_size=batch_size,
                                collate_fn=data_collator, shuffle=shuffle))

    def log_metric(self, metrics, step):
        # logger.info(metrics)
        # metrics = dict()
        # step = metrics.pop("step", -1)
        tag = metrics.pop("tag", "train")
        # import pdb;pdb.set_trace()
        assert step > 0, "未设置step"
        for k, v in metrics.items():
            self.writer.add_scalar(tag=f"{tag}/{k}",
                                   scalar_value=v,
                                   global_step=step)
    
    def clip_grad_norm_(self):
        if self.parameters == None:
            for v in self.model.values():
                self.parameters = [p for p in v.parameters() if p.requires_grad==True]
        torch.nn.utils.clip_grad_norm_(self.parameters, self.max_norm)

    def save_checkpoint(self, path):
        # for k, v in self.model.items():
        self.model["student"].save_pretrained(f"{path}/student_model")
        self.tokenizer.save_pretrained(f"{path}/student_model")
        # self.ffn.save_pretrained(f"{path}/ffn_model")
        pass
    
    def save_best_chpk(self, metrics):
        path = f"{self.saved_dir}/chpk-{self.update_step}"
        
        if self.metrics[0] not in metrics:
            socre = avg([v for k, v in metrics.items() if self.metrics[0] in k])
        else:
            socre = metrics[self.metrics[0]]
        if len(self.best_model) < 1 or len(self.best_model) < self.save_num:
            self.save_checkpoint(path)
            self.best_model.append({"chpk_path": path,
                           "metric": socre}) 
        elif socre > min([x["metric"] for x in self.best_model]):
            self.best_model.sort(key=lambda x: x["metric"])
            drop_model = self.best_model[0]
            self.best_model = self.best_model[1:]
            self.best_model.append({"chpk_path": path,
                                "metric": socre})
            if os.path.exists(drop_model["chpk_path"]):
                shutil.rmtree(drop_model["chpk_path"])
            self.save_checkpoint(path)

    def get_dataloader(self, shuffle):
        """构造data_loader 为后面step准备数据"""
        self.data_loader = dict()
        shuffle = self.shuffle
        for k, v in self.datasets.items():
            self.data_loader[k] = self._create_dataloader(v, self.batch_size, shuffle)
            logger.info(f"step {k} create dataloader on {v}")

        # return None

    def get_batch(self, split, shuffle):            ## TODO shuffle考虑去掉
        """

        """
        x = next(self.data_loader[split], None)
        if x is None:
            # get_datasets_iter()         #
            self.data_loader[split] = self._create_dataloader(self.datasets[split], self.batch_size, shuffle)
            logger.info(f"step {split} create dataloader on {self.datasets[split]}")
            # 待实现，就是如果一个epoch数据取完了，接下来创建新的数据集
            x = next(self.data_loader[split], None)
        assert x != None, "get x error"
        return x
    
    def train_step(self, train_steps_fn):
        loss = None                     # total loss
        return_metric = dict()

        x = self.get_batch(split="train", shuffle=self.shuffle)

        loss, return_metric = train_steps_fn(self.model, x)
        # outputs = self.translate_step(self.model["student_model"], x)

        # return_metric["translate_loss"] = loss.item()
        # loss = outputs.loss
        # for k_step, step_fn in train_steps:
        #     outputs = train_steps()

        # raise NotImplementedError

        return loss, return_metric

    def train_epoch_amp(self, scaler, train_steps_fn, p_bar, evaluate_step_fn=None):
        if evaluate_step_fn == None:
            evaluate_step_fn = evaluate_fn

        eval_update_num_epoch = self.epoch_size // self.accumulation
        eval_update_num_epoch = eval_update_num_epoch if self.epoch_size % self.accumulation == 0 else eval_update_num_epoch+1
        for step in range(self.epoch_size):
            with autocast():
                loss, metrics = self.train_step(train_steps_fn)
                loss = loss / self.accumulation             #
            scaler.scale(loss).backward()
            
            if step % self.accumulation == 0 or step+1 == self.epoch_size:  # 一个epoch结束的时候
                self.clip_grad_norm_()                       # 梯度裁剪

                scaler.step(self.optimizer)
                self.optimizer.zero_grad()
                self.update_step += 1
                p_bar.update(1)
                scaler.update()

                # log metrics
                if self.update_step % self.log_step == 0:
                    metrics["tag"] = "train"
                    metrics["epoch"] = self.update_step / eval_update_num_epoch
                    self.log_metric(metrics=metrics,step=self.update_step)
                # eval step
                if self.update_step % self.eval_step == 0:
                    metrics = evaluate_step_fn(self.model["student"], self.tokenizer,
                                          self.datasets.get("test", None), self.batch_size*2,
                                          self.num_beams, self.max_length, self.metrics,
                                          split="test", output_dir=self.saved_dir)
                    metrics["tag"] = "test"
                    self.log_metric(metrics=metrics, step=self.update_step)
                    for v in self.model.values():
                            v.train()                  
                ## ! 要改
                if self.update_step!=0 and (self.update_step % self.save_step == 0):
                    self.save_best_chpk(metrics)
        pass

    def train_end(self, epoch):
        logger.critical(f"训练结束，第{epoch} epoch {self.update_step} step 结束")

    def train(self, train_steps_fn, datasets, evaluate_fn=None, shuffle=True):
        """
        train_step 是一个函数，接收 所有的model, 和训练集中的x
        """
        assert self.model != None or self.tokenizer != None
        assert isinstance(self.model, dict)
        self.datasets = datasets
        self.get_dataloader(shuffle=shuffle)

        # 设置参数
        self.epoch_size = len(self.data_loader["train"])        # 有多少个批次，不是更新步数
        each_epoch_num_uptate = self.epoch_size // self.accumulation
        each_epoch_num_uptate = each_epoch_num_uptate + 1 if self.epoch_size % self.accumulation != 0 else each_epoch_num_uptate

        self.writer = SummaryWriter(self.saved_dir)         # 记录
        self.init_optimizer()

        if self.train_strategy == STRATEGY[1]:
            self.max_step = each_epoch_num_uptate* self.num_epoch
            assert self.eval_strategy == self.train_strategy and self.eval_step < 5
            self.eval_step = self.eval_step * each_epoch_num_uptate
            self.save_step = self.save_step * each_epoch_num_uptate
        self.update_step = 0

        # for k in dir(self):
        #     if not k.startswith("_") or not callable(getattr(self, k, None)):
        #         logger.info(f"{k} {getattr(self, k, None)}")
        
        if self.amp == True:
            scaler = GradScaler()
            p_bar = tqdm(total=self.max_step)                          ## TODO
            for epoch in range(self.num_epoch):
                self.train_epoch_amp(scaler, train_steps_fn, p_bar, evaluate_fn)
                ## ! 训练结束
                if self.update_step >= self.max_step:
                    break
            self.train_end(epoch)
            pass
        else:
            assert 1 == 2, "暂时为实现不适用混合精度的"
            pass
        pass




class TranslateTrainer(Trainer):

    def __init__(self, **args) -> None:
        super().__init__(**args)

    def check_params(self, args):
        super().check_params(args)

        pass

    

    


    def train_step(self):

        x = self.get_batch(split="train")
        x = to_same_device(x, self.model)

        outputs = self.model(**x)

        metrics = {"loss": outputs.loss}
        return outputs.loss, metrics, outputs

