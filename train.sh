
CUDA_VISIBLE_DEVICES="1"  python3 ./train.py \
    --src_lang nl_XX \
    --tgt_lang zh_CN \
    --bi false  \
    --student_path /data/hyxu/codes/cache_dir/mbart-large-cc25 \
    --num_beams 5 \
    --saved_dir /data/hyxu/codes/lowMT_compute/model \
    --label_smoothing_factor 0.2 \
    --lr 2e-5 \
    --batch_size 8 \
    --accumulation 4 \
    --max_norm 2 \
    --max_length 128   \
    --src_file 'train.nl'   \
    --tgt_file 'train.zh'   \
    --train_strategy epoch \
    --eval_strategy steps \
    --eval_step 3000 \
    --save_step 3000 \
    --log_step  200 \
    --num_epoch 10 \
    --warmup_steps 2000 \
    --steps "translate" \
    --optimer adamw \
    --metrics 'bleu,'\
    --data_dir /data/hyxu/codes/lowMT_compute/data/public_data/train/pair \
    --name 'ft-smooth-warmup'     \
# nohup ./train.sh > ./log/train_sm_warm.log 2>&1 &
# bsub -n 1 -q HPC.S1.GPU.X795.suda -o ./log/train_mbart50.log -gpu  num=1:mode=exclusive_process sh ./src/nmt_train.sh