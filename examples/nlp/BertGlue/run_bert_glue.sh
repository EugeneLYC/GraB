TASK=sst2

base_dir=`pwd`

python train_bert_glue.py \
  --model_name_or_path /home/user/transformer_ckpt/bertbase \
  --task_name ${TASK} \
  --train_file /home/user/data/GLUE/${TASK}/train.csv \
  --validation_file /home/user/data/GLUE/${TASK}/dev.csv \
  --max_length 128 \
  --per_device_train_batch_size 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --use_tensorboard \
  --tensorboard_path ${base_dir}/text-classification \
  --seed 0 \
  --shuffle_type dm