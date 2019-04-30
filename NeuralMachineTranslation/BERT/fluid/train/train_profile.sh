#!bin/bash
set -xe
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=/paddle/zcd_Paddle/build_fast/python/
export FLAGS_cudnn_deterministic=true
#export FLAGS_enable_parallel_graph=1
#export FLAGS_eager_delete_tensor_gb=0.0
#export FLAGS_fraction_of_gpu_memory_to_use=0.98
#export FLAGS_memory_fraction_of_eager_deletion=1.0
#export FLAGS_eager_delete_scope=false
task="BERT"
index=""
export GLOG_vmodule=parallel_executor=1
cd ./LARK_Paddle_BERT/BERT/
BERT_BASE_PATH=/ssd1/bert_data
TASK_NAME='XNLI'
DATA_PATH=${BERT_BASE_PATH}/data
CKPT_PATH=${BERT_BASE_PATH}/save

echo "CUDA_VISIBLE_DEVICES: " $CUDA_VISIBLE_DEVICES
echo "FLAGS_enable_parallel_graph: " $FLAGS_enable_parallel_graph
python -c 'import paddle;  print(paddle.__version__)'
python -c 'import paddle;  print(paddle.__git_commit__)'
echo "PYTHONPATH:" $PYTHONPATH

batch_size=32

python -u run_classifier_profile.py --task_name ${TASK_NAME} \
     --use_cuda true \
     --do_train true \
     --do_val true \
     --do_test true \
     --batch_size $batch_size \
     --in_tokens False \
     --init_pretraining_params ${BERT_BASE_PATH}/chinese_L-12_H-768_A-12/params \
     --data_dir ${DATA_PATH} \
     --vocab_path ${BERT_BASE_PATH}/chinese_L-12_H-768_A-12/vocab.txt \
     --checkpoints ${CKPT_PATH} \
     --save_steps 1000 \
     --shuffle false \
     --weight_decay  0.01 \
     --warmup_proportion 0.1 \
     --validation_steps 1000 \
     --epoch 2 \
     --max_seq_len 128 \
     --bert_config_path ${BERT_BASE_PATH}/chinese_L-12_H-768_A-12/bert_config.json \
     --learning_rate 5e-5 \
     --skip_steps 100 \
     --random_seed 1 

cd -
