PYTHON="/home/jmeng15/anaconda3/bin/python"
############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi
export CUDA_VISIBLE_DEVICES=3
num_runs=6
model=resnet50

dataset=sketches
data_path='/home/jmeng15/imagenet_to_sketch/sketches'
# data_path='/opt/imagenet/imagenet_compressed/'

epochs=30
batch_size=1
optimizer=SGD

# quantization scheme
channel_wise=0
wbit=4
abit=4

wd=1e-4
lr=5e-3

pretrained_model="./save/resnet50/resnet50_lr0.005_wd0.0001_channelwise0/model_best.pth.tar"

save_path="./save/${model}_CL/${dataset}/${model}_lr${lr}_wd${wd}_channelwise${channel_wise}/"
log_file="${model}_lr${lr}_wd${wd}_wbit${wbit}_abit${abit}_run${run_id}.log"

$PYTHON -W ignore main.py --dataset ${dataset} \
    --data_path ${data_path} \
    --model ${model} \
    --save_path ${save_path} \
    --log_file ${log_file} \
    --batch_size ${batch_size} \
    --ngpu 1 \
    --wbit ${wbit} \
    --abit ${abit} \

    
