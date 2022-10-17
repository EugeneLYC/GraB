model=logistic_regression
dataset=mnist
epochs=100
bsz=2
gradaccstep=32
lr=0.01
stype=dm

base_dir=`pwd`

run_cmd="python train_logreg_mnist.py --model=${model} \
        --dataset=${dataset} \
        --data_path=${base_dir} \
        --epochs=${epochs} \
        --batch_size=${bsz} \
        --grad_accumulation_step=${gradaccstep} \
        --lr=${lr} \
        --shuffle_type=${stype} \
        --momentum=0.9 \
        --weight_decay=1e-4 \
        --use_tensorboard \
        --tensorboard_path=${base_dir} \
        --num_workers=0
        "

echo ${run_cmd}
eval ${run_cmd}
