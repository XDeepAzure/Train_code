CUDA_VISIBLE_DEVICES="7" python3 ./eval.py \
    --src_lang nl_XX \
    --tgt_lang zh_CN \
    --model_path /data/hyxu/lowMT_compute/model/ft-smo-warm/chpk-20000/model  \
    --output_dir /data/hyxu/lowMT_compute/model/ft-smo-warm/chpk-20000/model \
    --batch_size 24 \
    --max_length 128   \
    --num_beams 4   \
    --seed 10 \
    --data_dir '/data/hyxu/lowMT_compute/data/public_data/' \
    --src_file test.zh \
    --tgt_file test.en \
    --metrics 'bleu,'  \
    --split test \
    --save_text true \
    --test_dataset dev_set \

# nohup ./eval.sh > ./log/eval.log 2>&1 &
# /data/hyxu/cached_dir/nllb-200-distilled-600M