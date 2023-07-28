
CUDA_VISIBLE_DEVICES="2"  python3 ./train.py \
    --src_lang nl_XX \
    --tgt_lang zh_CN \
    --bi false  \
    --student_path /data/hyxu/cached_dir/mbart-large-cc25 \
    --num_beams 5 \
    --saved_dir /data/hyxu/lowMT_compute/model \
    --label_smoothing_factor 0 \
    --lr 2e-5 \
    --batch_size 16 \
    --accumulation 4 \
    --max_norm 2 \
    --max_length 128   \
    --src_file 'train.nl-zh.nl'   \
    --tgt_file 'train.nl-zh.zh'   \
    --denoising_langs "nl_XX,zh_CN" \
    --denoising_file "clean.train.nl-zh.nl,clean.train.nl-zh.zh"  \
    --train_strategy epoch \
    --eval_strategy steps \
    --eval_step 2000 \
    --save_step 2000 \
    --log_step  200 \
    --num_epoch 30 \
    --warmup_steps 2000 \
    --steps "translate,denoising" \
    --w_noise 0.5 \
    --optimer adamw \
    --metrics 'bleu,'\
    --data_dir /data/hyxu/lowMT_compute/data/public_data/train/pair \
    --name 'ft-denoise1'     \
# nohup ./train.sh > ./log/train_noise1.log 2>&1 &
# bsub -n 1 -q HPC.S1.GPU.X795.suda -o ./log/train_mbart50.log -gpu  num=1:mode=exclusive_process sh ./src/nmt_train.sh