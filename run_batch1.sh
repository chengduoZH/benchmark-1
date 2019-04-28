

export PYTHONPATH=/paddle/zcd_Paddle/build_fast/python
 
export task_id=0 # origin task

cd NeuralMachineTranslation/BERT/fluid/train
export CUDA_VISIBLE_DEVICES=7
sh ./train_debug.sh   > ../../../../perf_log/result_${task_id}_BERT_1.txt 2>&1
cd -

export PYTHONPATH=/paddle/dev_Paddle/build_fast/python
 
export task_id=1 # origin task

cd NeuralMachineTranslation/BERT/fluid/train
export CUDA_VISIBLE_DEVICES=7
sh ./train_debug.sh   > ../../../../perf_log/result_${task_id}_BERT_1.txt 2>&1
cd -


