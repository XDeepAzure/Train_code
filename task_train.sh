
steps=$1
lr=$2
w_enc=$3
name=$4

CUDA_VISIBLE_DEVICES="3,6"  python3 ./train.py \
    --src_lang tur_Latn \
    --tgt_lang eng_Latn \
    --student_path /data/hyxu/cached_dir/nllb-200-distilled-600M \
    --teacher_path /data/hyxu/cached_dir/nllb-200-distilled-1.3B \
    --num_beams 4  \
    --saved_dir /data/hyxu/Pruner/model \
    --lr $lr \
    --batch_size 16 \
    --accumulation 4 \
    --max_norm 2 \
    --max_length 128   \
    --src_file 'wmt17.en-tr.train.tr,tur_Latn'   \
    --tgt_file 'wmt17.en-tr.train.en,eng_Latn'   \
    --train_strategy epoch \
    --eval_strategy epoch \
    --eval_step 1 \
    --save_step 1 \
    --log_step  100 \
    --steps $steps \
    --w_enc $w_enc   \
    --w_dec 20  \
    --optimer adamw \
    --metrics 'chrf,'\
    --data_dir /data/hyxu/Pruner/data/nllb-seed-dataset/train  \
    --name $name     \
# nohup ./train.sh > ./log/train.log 2>&1 &
# bsub -n 1 -q HPC.S1.GPU.X795.suda -o ./log/train_mbart50.log -gpu  num=1:mode=exclusive_process sh ./src/nmt_train.sh