from datetime import timedelta
import os
import random
import time
import numpy as np
from tqdm import tqdm
import json
import torch
import logging

langs = ['nus_Latn', 'ltg_Latn', 'arz_Arab', 'srd_Latn', 'mag_Deva', 'bjn_Latn', 'ace_Arab', \
         'ary_Arab', 'knc_Arab', 'ban_Latn', 'tzm_Tfng', 'fuv_Latn', 'fur_Latn', 'shn_Mymr', \
         'bug_Latn', 'taq_Tfng', 'bam_Latn', 'prs_Arab', 'taq_Latn', 'kas_Arab', 'crh_Latn', \
         'dzo_Tibt', 'lij_Latn', 'hne_Deva', 'szl_Latn', 'vec_Latn', 'grn_Latn', 'knc_Latn', \
         'dik_Latn', 'lmo_Latn', 'ace_Latn', 'pbt_Arab', 'lim_Latn', 'kas_Deva', 'bjn_Arab', \
         'mri_Latn', 'bho_Deva', 'scn_Latn', 'mni_Beng', "eng_Latn"]

# def setup_seed(seed):
#     torch.manual_seed(seed)                                 #不是返回一个生成器吗？
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True               #使用确定性的卷积算法
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''

def create_logger(filepath=None, rank=0, name=None):
    """
    Create a logger.
    Use a different log file for each process.
    filepath 为None的时候即不输出到文本里面去，
    rank为0的时候即单线程
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = '%s-%i' % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    if name != None:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger