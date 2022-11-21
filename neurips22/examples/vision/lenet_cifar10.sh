model=lenet
dataset=cifar10
epochs=100
bsz=8
gradaccstep=2
lr=0.001
stype=dm

base_dir=`pwd`

run_cmd="python train_lenet_cifar.py --model=${model} \
        --dataset=${dataset} \
        --data_path=${base_dir} \
        --epochs=${epochs} \
        --batch_size=${bsz} \
        --grad_accumulation_step=${gradaccstep} \
        --lr=${lr} \
        --shuffle_type=${stype} \
        --momentum=0.9 \
        --weight_decay=1e-2 \
        --use_tensorboard \
        --tensorboard_path=${base_dir} \
        --num_workers=0
        "

echo ${run_cmd}
eval ${run_cmd}
