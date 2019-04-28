

export PYTHONPATH=/paddle/zcd_Paddle/build_fast/python
 
task_id= 0 # origin task


cd CycleGAN/paddle/
sh ./train_debug.sh  > ../../perf_log/result_${task_id}_cycle_gan.txt 2>&1
cd -

cd NeuralMachineTranslation/Transformer/fluid/train/
export CUDA_VISIBLE_DEVICES=7
sh ./train_debug.sh   > ../../../../perf_log/result_${task_id}_transformer_1.txt 2>&1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
sh ./train_debug.sh  > ../../../../perf_log/result_${task_id}_transformer_8.txt 2>&1
cd -

cd NeuralMachineTranslation/BERT/fluid/train
export CUDA_VISIBLE_DEVICES=7
sh ./train_debug.sh   > ../../../../perf_log/result_${task_id}_BERT_1.txt 2>&1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
sh ./train_debug.sh  > ../../../../perf_log/result_${task_id}_BERT_8.txt 2>&1
cd -

cd PaddingRNN/lstm_paddle
export CUDA_VISIBLE_DEVICES=7
sh ./train_debug.sh   > ../../../../perf_log/result_${task_id}_PaddingRnn_1.txt 2>&1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
sh ./train_debug.sh  > ../../../../perf_log/result_${task_id}_PaddingRnn_8.txt 2>&1
cd -

