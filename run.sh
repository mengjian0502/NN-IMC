PYTHON="/home/li/.conda/envs/neurosim_test/bin/python"
############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi
export CUDA_VISIBLE_DEVICES=1
model=resnet50

dataset=wikiart
data_path='/home/li/data/imagenet_to_sketch/wikiart'
# data_path='/opt/imagenet/imagenet_compressed/'

batch_size=1

# quantization scheme
channel_wise=0
wbit=4
abit=4

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
    --fine_tune

    
