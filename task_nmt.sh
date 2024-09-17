
file=$1
gpu="3"
echo "gpu is ${gpu}"
CUDA_VISIBLE_DEVICES=$gpu  python3 ./train.py \
    --src_lang nl_XX \
    --tgt_lang zh_CN \
    --bi false  \
    --student_path /data/hyxu/lowMT_compute/model/bt-smo/chpk-8000/model \
    --num_beams 5 \
    --saved_dir /data/hyxu/lowMT_compute/model \
    --label_smoothing_factor 0.2 \
    --lr 2e-5 \
    --batch_size 16 \
    --accumulation 4 \
    --max_norm 2 \
    --max_length 128   \
    --src_file 'mine.nl-zh.nl'   \
    --tgt_file 'mine.nl-zh.zh'   \
    --denoising_langs "nl_XX,zh_CN" \
    --denoising_file "clean.train.nl-zh.nl,clean.train.nl-zh.zh"  \
    --train_strategy epoch \
    --eval_strategy epoch \
    --eval_step 1 \
    --save_step 1 \
    --log_step  400 \
    --num_epoch 5 \
    --warmup_steps 0 \
    --steps "translate" \
    --w_noise 0.2 \
    --beta 0 \
    --optimer adamw \
    --metrics 'bleu,'\
    --data_dir /data/hyxu/lowMT_compute/data/public_data/train/pair/ \
    --name 'bt-smo-mine'     \
# nohup ./nmt.sh > ./log/mine.log 2>&1 &
# bsub -n 1 -q HPC.S1.GPU.X795.suda -o ./log/train_mbart50.log -gpu  num=1:mode=exclusive_process sh ./src/nmt_train.sh