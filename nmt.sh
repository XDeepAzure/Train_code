gpu="1"
echo "gpu is ${gpu}"
CUDA_VISIBLE_DEVICES=$gpu  python3 ./train.py \
    --src_lang nl_XX \
    --tgt_lang zh_CN \
    --bi false  \
    --student_path /data/hyxu/codes/lowMT_compute/model/ft-mine/chpk-166/model \
    --num_beams 5 \
    --saved_dir /data/hyxu/codes/lowMT_compute/model \
    --label_smoothing_factor 0.2 \
    --lr 1e-5 \
    --batch_size 16 \
    --accumulation 4 \
    --max_norm 2 \
    --max_length 128   \
    --src_file 'bt.nl'   \
    --tgt_file 'bt.zh'   \
    --denoising_langs "nl_XX,zh_CN" \
    --denoising_file "clean.train.nl-zh.nl,clean.train.nl-zh.zh"  \
    --train_strategy epoch \
    --eval_strategy steps \
    --eval_step 2000 \
    --save_step 2000 \
    --log_step  400 \
    --num_epoch 4 \
    --warmup_steps 0 \
    --steps "translate" \
    --w_noise 0.2 \
    --beta 0 \
    --optimer adamw \
    --metrics 'bleu,'\
    --data_dir /data/hyxu/codes/lowMT_compute/data/public_data/train/pair/bt-pair \
    --name 'bt-smo'     \
# nohup ./nmt.sh > ./log/bt.log 2>&1 &
# bsub -n 1 -q HPC.S1.GPU.X795.suda -o ./log/train_mbart50.log -gpu  num=1:mode=exclusive_process sh ./src/nmt_train.sh