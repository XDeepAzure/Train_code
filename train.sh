
CUDA_VISIBLE_DEVICES="0"  python3 ./train.py \
    --src_lang nl_XX \
    --tgt_lang zh_CN \
    --bi false  \
    --student_path /data/hyxu/codes/cache_dir/mbart-large-cc25 \
    --num_beams 5 \
    --saved_dir /data/hyxu/codes/lowMT_compute/model \
    --label_smoothing_factor 0 \
    --lr 2e-4 \
    --batch_size 16 \
    --accumulation 4 \
    --max_norm 2 \
    --max_length 128   \
    --src_file 'train.nl'   \
    --tgt_file 'train.zh'   \
    --train_strategy epoch \
    --eval_strategy steps \
    --eval_step 4000 \
    --save_step 4000 \
    --log_step  400 \
    --num_epoch 50 \
    --warmup_steps 3000 \
    --steps "translate" \
    --optimer adamw \
    --metrics 'bleu,'\
    --data_dir /data/hyxu/codes/lowMT_compute/data/public_data/train/pair \
    --name 'ft-warmup'     \
# nohup ./train.sh > ./log/train_warm.log 2>&1 &
# bsub -n 1 -q HPC.S1.GPU.X795.suda -o ./log/train_mbart50.log -gpu  num=1:mode=exclusive_process sh ./src/nmt_train.sh