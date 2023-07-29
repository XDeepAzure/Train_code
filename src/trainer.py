import os
from logging import getLogger
import shutil
from dataclasses import dataclass
import torch

from tqdm import tqdm
from transformers import (AutoTokenizer, DataCollatorForSeq2Seq)

from torch.utils.data import DataLoader

# 自动混合精度
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

import torch.nn as nn
import torch.nn.functional as F
import pytorch_warmup as warmup

from src.loss import In_trust_Loss

from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder, M2M100Model, shift_tokens_right

from torch.utils.tensorboard import SummaryWriter

logger = getLogger()




OPTIMIZER = ("AdamW", "SGD")
STRATEGY = ("steps", "epoch", "no")
STEPS = ("translate", "denoising", "enc", "dec", "enc-dec", "dec_noise")

def avg(x):
    # # assert isinstance(x[0], int)
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

def translate_step(model, x, output_hidden_states=False):
    """ 翻译任务的步骤，包含 encoder_last_hidden_state,  decoder_hidden_states(12层)"""
    for k, v in x.items():
        x[k] = v.to(model.device)
    outputs = model(**x, output_hidden_states=output_hidden_states)
    return {k: v for k, v in outputs.items()}

def denoising_step(model, x, lang, w):
    for k, v in x.items():
        x[k] = v.to(model.device)
    outputs = model(**x, output_hidden_states=False)
    return {"loss": outputs.loss * w, f"{lang}_denoising_loss": outputs.loss.item()}

def r_drop_step(model, x, w):
    """对token粒度做不太好"""
    feature = model.model.encoder(x["input_ids"], x["attention_mask"]) 
    feature_2 = model.model.encoder(x["input_ids"], x["attention_mask"])
    kl_loss1 = F.kl_div(F.log_softmax(feature, dim=-1), F.softmax(feature_2, dim=-1))
    kl_loss2 = F.kl_div(F.log_softmax(feature_2, dim=-1), F.softmax(feature, dim=-1))
    kl_loss = (kl_loss1 + kl_loss2) / 2
    return {"loss": w * kl_loss, "r_drop_loss": kl_loss.item()}

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

def tsda_step(model, outputs, inputs, w, mix_ratio=0.2, temperature=2):
    """https://github.com/TARGET-SIDE-DATA-AUG/TSDASG/blob/main/fairseq/fairseq/models/transformer.py
    数据增强策略
    """
    # import pdb;pdb.set_trace()
    # shift right
    x = outputs["logits"].clone()
    length = len(x[0])
    for idx in range(length - 1, -1, -1):
        x[:,idx] = x[:,idx - 1]     
            
    # set the second index of logits as the maximum so that after softmax, the first token must be '2'
    # The reason we set first token as '2' is that the first token of 'prev_output_tokens' is 2.
    x[:,0, 2] = 2 * torch.max(x[:,0])   
    x = F.softmax(x / temperature, dim = 2)

    # mask pad, pad = 1. We found this is unnecessary.
    #x = x.masked_fill(prev_output_tokens.eq(1),  1)
    
    # Make the output dimension the same as the input
    with torch.no_grad():     
        embed_matrix = model.model.decoder.embed_tokens.weight.clone()   # vocab_size * embed_lenghth (10152 * 512)        
        decoder_in = torch.einsum('blv,ve->ble', x, embed_matrix) # batch * len * embed_lenghth
        
    # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
    decoder_outputs = model.model.decoder(
        inputs_embeds=decoder_in,
        attention_mask=inputs["labels"] != -100,
        encoder_hidden_states=outputs["encoder_last_hidden_state"],
        encoder_attention_mask=inputs["attention_mask"],
    )

    logits = model.lm_head(decoder_outputs[0]) + model.final_logits_bias
    
    loss_fct = nn.CrossEntropyLoss()
    loss2 = loss_fct(logits.view(-1, model.config.vocab_size), inputs["labels"].view(-1))
        
    # We use KL divergence here
    P = F.log_softmax(logits,dim = 2)
    Q = F.softmax(outputs["logits"], dim=2)
    dk = F.kl_div(P, Q, reduction='sum')

    loss_total_without_kd = loss2 * mix_ratio + dk
    return {"loss": w * loss_total_without_kd, "tsda_loss": loss_total_without_kd.item()}


def contrast_step(model, outputs, x, w):
    '''list_wise loss'''

    output_features = self.get_global_features(output_hidden_states)
    label_features = self.get_global_features(label_hidden_states)
        

    # 获取正例相似度和负例相似度
    pos_similarities = F.cosine_similarity(output_features, label_features,dim=-1)  #torch.Size([48, 1])
    neg_similarities = F.cosine_similarity(output_features.transpose(0,1), label_features,dim=-1)  #torch.Size([48, 48])
    # 计算对比损失函数
    temperature = 0.3  # InfoNCE损失函数的温度参数
    # cl_loss = -torch.mean(torch.log(torch.exp(pos_similarities/temperature) /  
    #                        torch.sum(torch.exp(neg_similarities/temperature))))
    
    logits = neg_similarities
    labels = torch.arange(len(label_features),device=label_features.device)
    cl_loss = F.cross_entropy(logits / temperature, labels)
    pass

class Trainer(object):

    def __init__(self, model=None, tokenizer=None, accumulation=1, max_norm=2, max_length=128, num_beams=4,
                 batch_size=32, seed=10, saved_dir="./", shuffle=True, datasets=None,
                 train_strategy=STRATEGY[0], eval_strategy=STRATEGY[0], save_step=1, eval_step=1, log_step=100,
                 num_epoch=3, max_step=10000, metrics=["chrf"], amp=True, optimizer=OPTIMIZER[0], lr=2e-5,
                 weight_decay=0.01, save_num=3, label_smoothing_factor=0, warmup_steps=2000,
                 alpha=1, beta=0, delta=0.5) -> None:

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
        if self.eval_strategy != STRATEGY[2]:
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
        self.warmup_steps = warmup_steps

        self.alpha = alpha                      # 是否降噪学习的loss
        self.beta = beta                        # 若不为0 则为开启
        self.delta = delta  
        
        self.parameters = None
        self.best_model = []
        self.save_num = save_num
        self.update_step = 0

        self.label_smoothing_factor = label_smoothing_factor

        pass

    def check_params(self, args):
        """设置一些默认值"""
        
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
        """可以添加上warm up的方法"""
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

    def create_lr_scheduler(self):
        if self.optimizer == None:
            self.init_optimizer()
        if self.warmup_steps <= 0:
            start_factor, end_factor = 1.0, 1.0
            total_iters: int = 1
            last_epoch = -1
        else:
            start_factor, end_factor= 0.3, 1.0
            total_iters: int = self.warmup_steps
            last_epoch = -1
        
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=start_factor, end_factor=end_factor,
                                                               total_iters=total_iters, last_epoch=last_epoch, verbose=False)

    def _create_dataloader(self, dataset, batch_size, shuffle=False):
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, padding=True, 
                                               max_length=self.max_length)

        return iter(DataLoader(dataset, batch_size=batch_size,
                                collate_fn=data_collator, shuffle=shuffle))

    def log_metric(self, metrics, step):
        tag = metrics.pop("tag", "train")
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
        self.model["model"].save_pretrained(f"{path}/model")
        self.tokenizer.save_pretrained(f"{path}/model")
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
            ## ! 扩展一下到多个数据集
            if not isinstance(v, dict):
                self.data_loader[k] = self._create_dataloader(v, self.batch_size, shuffle)
                logger.info(f"step {k} create dataloader on {v}")
            else:
                dit = dict()
                for sub_k, sub_v in v.items():
                    dit[sub_k] = self._create_dataloader(sub_v, self.batch_size, shuffle)
                    logger.info(f"step {k} {sub_k} create dataloader on {sub_v}")
                self.data_loader[k] = dit


        # return None

    def get_batch(self, step, split, shuffle):            ## TODO shuffle考虑去掉
        """

        """
        x = next(self.data_loader[step][split], None)
        if x is None:
            # get_datasets_iter()         #
            self.data_loader[step][split] = self._create_dataloader(self.datasets[step][split], self.batch_size, shuffle)
            logger.info(f"{step} {split} create dataloader on {self.datasets[step][split]}")
            # 待实现，就是如果一个epoch数据取完了，接下来创建新的数据集
            x = next(self.data_loader[step][split], None)
        assert x != None, "get x error"
        return x
    
    def label_smooth_step(self, outputs, labels, shift_labels=False):
        return_metrics = {"translate_loss": outputs["loss"].item() if isinstance(outputs, dict) else outputs.loss.item()}
        if self.label_smoother:
            loss = self.label_smoother(outputs, labels, shift_labels=shift_labels)
            if hasattr(outputs, "loss"):
                outputs.loss = loss
            else:
                outputs["loss"] = loss
            return_metrics["smooth_loss"] = outputs.loss.item() if hasattr(outputs, "loss") else outputs["loss"].item()
        return_metrics["loss"] = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        return return_metrics

    def _create_Intrust_loss(self):
        if 1 > self.beta > 0:
            assert self.label_smoother == None, "label smooth 不能与 in trust loss 同时用"
            logger.critical("使用抗噪学习的loss")
            self.in_truct_loss = In_trust_Loss(alpha=self.alpha, beta=self.beta, delta=self.delta,
                                               num_classes=self.tokenizer.vocab_size)
        else:
            self.in_truct_loss = None
            pass
    
    def in_trust_loss_step(self, outputs, labels):
        """抗噪loss"""
        return_metrics = {"translate_loss": outputs["loss"].item()}
        if self.in_truct_loss:
            logits = outputs["logits"].view(-1, self.in_truct_loss.num_classes) 
            labels = labels.view(-1)
            loss = self.in_truct_loss(logits, labels)
            outputs["loss"] = loss
        else:
            pass
        return_metrics["loss"] = outputs["loss"]
        return return_metrics

    def post_step(self, outputs, labels):
        if self.label_smoother:
            return self.label_smooth_step(outputs, labels, shift_labels=False)
        elif self.in_truct_loss:
            return self.in_trust_loss_step(outputs, labels)
        else:
            return {"loss": outputs["loss"], "translate_loss": outputs["loss"].item()}

    def train_step(self, train_steps_fn):
        loss = None                     # total loss
        return_metric = dict()

        # ! 数据调用
        # x = self.get_batch(step="translate", split="train", shuffle=self.shuffle)

        loss, return_metric = train_steps_fn(self.model)

        return loss, return_metric

    def train_epoch_amp(self, scaler, train_steps_fn, p_bar, evaluate_step_fn=None):
        if evaluate_step_fn == None:
            evaluate_step_fn = evaluate_fn

        eval_update_num_epoch = self.epoch_size // self.accumulation
        eval_update_num_epoch = eval_update_num_epoch if self.epoch_size % self.accumulation == 0 else eval_update_num_epoch+1
        for step in range(self.epoch_size):
            with autocast():
                outputs = self.train_step(train_steps_fn)
                loss = 0
                return_metrics = dict()
                for m in outputs:
                    loss += m.pop("loss")
                    for k, v in m.items():
                        if isinstance(v, float) or (isinstance(v, torch.tensor) and len(v.shape) == 1 and v.shape[0]==1):
                            return_metrics[k] = v
                loss = loss / self.accumulation             #
            
            scaler.scale(loss).backward()
            
            if step % self.accumulation == 0 or step+1 == self.epoch_size:  # 一个epoch结束的时候
                self.clip_grad_norm_()                       # 梯度裁剪

                scaler.step(self.optimizer)
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                self.update_step += 1
                p_bar.update(1)
                scaler.update()

                # log metrics
                if self.update_step % self.log_step == 0:
                    metrics["tag"] = "train"
                    metrics["epoch"] = self.update_step / eval_update_num_epoch
                    metrics["lr"] = self.optimizer.param_groups[0]['lr']
                    self.log_metric(metrics=metrics,step=self.update_step)
                
                # eval step
                if self.update_step % self.eval_step == 0:
                    metrics = evaluate_step_fn(self.model, self.tokenizer,
                                          self.datasets, self.batch_size,
                                          self.num_beams, self.max_length, self.metrics,
                                          split="dev", output_dir=self.saved_dir)
                    metrics["tag"] = "dev"
                    self.log_metric(metrics=metrics, step=self.update_step)
                    for v in self.model.values():
                            v.train()                  
                ## ! 要改
                if self.update_step!=0 and (self.update_step % self.save_step == 0):
                    if self.eval_strategy == STRATEGY[2]:
                        self.save_checkpoint(f"{self.saved_dir}/chpk-{self.update_step}")
                    else:
                        self.save_best_chpk(metrics)
        pass

    def train_end(self, epoch):
        # self.writer.add_hparams(hparam_dict={},
        #                         metric_dict={self.metrics[0]: self.best_model["metric"]})
        logger.critical(f"训练结束，第{epoch} epoch {self.update_step} step 结束")
        logger.critical(f"best model is {self.best_model}")
        if len(self.best_model) < 1:
            self.save_checkpoint(path = f"{self.saved_dir}/chpk-{self.update_step}")

    def train(self, train_steps_fn, datasets, evaluate_fn=None, shuffle=True):
        """
        train_step 是一个函数，接收 所有的model, 和训练集中的x
        """
        assert self.model != None or self.tokenizer != None
        assert isinstance(self.model, dict)
        self.datasets = datasets
        self.get_dataloader(shuffle=shuffle)

        if self.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.label_smoothing_factor)
        else:
            self.label_smoother = None

        # 设置参数
        self.epoch_size = len(self.data_loader[STEPS[0]]["train"]) if STEPS[0] in self.data_loader \
                            else len(self.data_loader[STEPS[1]]["nl_XX"])
        each_epoch_num_uptate = self.epoch_size // self.accumulation
        each_epoch_num_uptate = each_epoch_num_uptate + 1 if self.epoch_size % self.accumulation != 0 else each_epoch_num_uptate

        self._create_Intrust_loss()
        self.writer = SummaryWriter(self.saved_dir)         # 记录
        self.init_optimizer()
        self.create_lr_scheduler()

        if self.train_strategy == STRATEGY[1]:
            self.max_step = each_epoch_num_uptate* self.num_epoch
        if self.eval_strategy == STRATEGY[1]:
            self.eval_step = self.eval_step * each_epoch_num_uptate
            self.save_step = self.save_step * each_epoch_num_uptate
        elif self.eval_strategy == STRATEGY[2]:
            self.eval_step = self.max_step + 1
        self.update_step = 0
        
        if self.amp == True:
            scaler = GradScaler()
            p_bar = tqdm(total=self.max_step)                          ## TODO
            for epoch in range(self.num_epoch):
                logger.info(f"第 {epoch} 个epoch训练开始")
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

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss



"""sdgfadsgsedrgfws"""
# Label smoothing
        # if self.args.label_smoothing_factor != 0:
        #     self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        # else:
        #     self.label_smoother = None

@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels, shift_labels=False):
        # mbartForCodication 是false
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)                         # ignore_index
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss




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
