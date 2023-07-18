import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # 模型相关
    parser.add_argument(
        "--src_lang", type=str, default="", help="tokenizer 中设置的语言代码"
    )
    parser.add_argument(
        "--tgt_lang", type=str, default="eng_Latn"
    )
    parser.add_argument(
        "--student_path", type=str, default="", help="student model 所在位置"
    )
    parser.add_argument(
        "--bi", type=lambda x: x=="true", default=False, help="student model 所在位置"
    )
    parser.add_argument(
        "--max_length", type=int, default=256, help="训练中的句子最大长度"
    )
    parser.add_argument(
        "--max_generate_length", type=int, default=256, help="生成中的最大长度"
    )
    parser.add_argument(
        "--num_beams", type=int, default=0, help="暂时不改动"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="", help="默认为model路径"
    )
    parser.add_argument(
        "--data_collator", type=str, default="", help="默认不使用"
    )
    parser.add_argument(                            # 暂时不用
        "--pretrained_model", type=str, default="/public/home/hongy/pre-train_model", help="预训练模型的位置" 
    )
    ## ! 蒸馏相关设置
    ## ! 数据集
    parser.add_argument(
        "--data_dir", type=str, default="task/deep-encoder-shallow-decoder/data/en-ur", help="数据存放的位置"
    )
    parser.add_argument(
        "--tokenized_datasets", type=str, default="", help="如果为空，下面的file才起作用"
    )
    parser.add_argument(
        "--src_file", type=str, default="", help="必须包括valid的, 用','分隔"
    )
    parser.add_argument(
        "--tgt_file", type=str, default=""
    )
    parser.add_argument(
        "--test_dataset", type=str, default="flores", help="在train和retrain里决定是否用flores的dev和test集"
    )
    ## !评估
    parser.add_argument(
        "--metrics", type=str, default="chrf", help="用,分隔开"
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=15, help="暂时不设"
    )
    ## ! trainer相关
    parser.add_argument(
        "--num_epoch", type=int, default=3
    )
    parser.add_argument(                        # 暂未实现
        "--resume_from_checkpoint", type=str, default="", help="如果有那个模型训练中断了的话，用此参数来重新加载checkpoints继续训练"
    )
    parser.add_argument(
        "--optimer", type=str, default="adamw", help="优化器"
    )
    parser.add_argument(                        # 暂未实现
        "--warmup_steps", type=int, default=100
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="训练baseline用的是2e-5, retrain用的是4e-6"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="默认是16,"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=40, help="默认是40,"
    )
    parser.add_argument(
        "--saved_dir", type=str, default="", help="数据或者模型 或者评估结果 保存的位置"
    )
    parser.add_argument(
        "--train_strategy", type=str, default="", help="数据或者模型 或者评估结果 保存的位置"
    )
    parser.add_argument(
        "--eval_strategy", type=str, default="", help="数据或者模型 或者评估结果 保存的位置"
    )
    parser.add_argument(
        "--steps", type=str, default="translate,", help="训练哪些部分"
    )
    parser.add_argument(
        "--w_enc", type=str, default=5, help="各种任务的权重与steps对应的"
    )
    parser.add_argument(
        "--w_dec", type=int, default=20, help="各种任务的权重与steps对应的"
    )
    parser.add_argument(
        "--w_noise", type=str, default=1, help="各种任务的权重与steps对应的"
    )
    parser.add_argument(
        "--max_step", type=int, default=2000
    )
    parser.add_argument(
        "--eval_step", type=int, default=2000
    )
    parser.add_argument(
        "--save_step", type=int, default=2000
    )
    parser.add_argument(
        "--log_step", type=int, default=250
    )
    parser.add_argument(
        "--accumulation", type=int, default=2, help="与batch_size组合使用,一般默认为2"
    )
    parser.add_argument(
        "--max_norm", type=int, default=2, help=""
    )
    parser.add_argument(                            # 暂未实现
        "--label_smoothing_factor", type=float, default=0
    )
    parser.add_argument(
        "--seed", type=int, default=10
    )
    parser.add_argument(
        "--shuffle", type=lambda x: x=="false", default=True
    )
    parser.add_argument(
        "--name", type=str, default="cor10w", help="标识此实验是干嘛的"
    )
    parser.add_argument(
        "--des", type=str, default="", help="写进log里描述再干什么"
    )
    args = parser.parse_args()
    return args
