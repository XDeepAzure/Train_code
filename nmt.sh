
CUDA_VISIBLE_DEVICES="0"  python3 ./train.py \
    --src_lang nl_XX \
    --tgt_lang zh_CN \
    --bi false  \
    --student_path /data/hyxu/codes/cache_dir/mbart-large-cc25 \
    --num_beams 5 \
    --saved_dir /data/hyxu/codes/lowMT_compute/model \
    --label_smoothing_factor 0.2 \
    --lr 2e-5 \
    --batch_size 16 \
    --accumulation 4 \
    --max_norm 2 \
    --max_length 128   \
    --src_file 'train.nl'   \
    --tgt_file 'train.zh'   \
    --denoising_langs "nl_XX,zh_CN" \
    --denoising_file "train.nl,train.zh"  \
    --train_strategy epoch \
    --eval_strategy epoch \
    --eval_step 1 \
    --save_step 1 \
    --log_step  400 \
    --num_epoch 20 \
    --warmup_steps 0 \
    --steps "translate" \
    --w_noise 0.5 \
    --beta 0 \
    --optimer adamw \
    --metrics 'bleu,'\
    --data_dir /data/hyxu/codes/lowMT_compute/data/public_data/train/pair \
    --name 'ft-clean-smo'     \
# nohup ./nmt.sh > ./log/ft-clean-smo.log 2>&1 &
# bsub -n 1 -q HPC.S1.GPU.X795.suda -o ./log/train_mbart50.log -gpu  num=1:mode=exclusive_process sh ./src/nmt_train.sh