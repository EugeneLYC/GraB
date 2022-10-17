base_dir=`pwd`
python3 main.py --shuffle_type dm \
    --data ./wikitext-2 \
    --bptt 35 \
    --lr 5 \
    --seed 1 \
    --use_tensorboard \
    --tensorboard_path ${base_dir}