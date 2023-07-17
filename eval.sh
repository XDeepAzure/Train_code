CUDA_VISIBLE_DEVICES="1" python3 ./eval.py \
    --src_lang tur_Latn \
    --tgt_lang eng_Latn \
    --model_path /data/hyxu/cached_dir/nllb-200-distilled-600M  \
    --output_dir /data/hyxu/Pruner/model \
    --batch_size 32 \
    --max_length 128   \
    --num_beams 4   \
    --data_dir '/public/home/hongy/Translation/data/dataset/BWB/processed'\
    --src_file test.zh \
    --tgt_file test.en \
    --metrics 'chrf,'  \
    --multi_language both \
    --split test \
    --test_dataset flores \

# nohup ./eval.sh > ./log/eval.log 2>&1 &
# /data/hyxu/cached_dir/nllb-200-distilled-600M