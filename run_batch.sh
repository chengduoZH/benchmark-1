

export PYTHONPATH=/paddle/zcd_Paddle/build_fast/python
 
cd NeuralMachineTranslation/BERT/fluid/train
export CUDA_VISIBLE_DEVICES=7
sh ./train_debug.sh   > ../../../../perf_log/result_BERT_1.txt 2>&1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
sh ./train_debug.sh  > ../../../../perf_log/result_BERT_8.txt 2>&1
cd -


cd CycleGAN/paddle/
sh ./train_debug.sh  > ../../perf_log/result_cycle_gan.txt 2>&1
cd -


cd NeuralMachineTranslation/Transformer/fluid/train/
export CUDA_VISIBLE_DEVICES=7
sh ./train_debug.sh   > ../../../../perf_log/result_transformer_1.txt 2>&1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
sh ./train_debug.sh  > ../../../../perf_log/result_transformer_8.txt 2>&1
cd -

cd NeuralMachineTranslation/BERT/fluid/train
export CUDA_VISIBLE_DEVICES=7
sh ./train_debug.sh   > ../../../../perf_log/result_BERT_1.txt 2>&1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
sh ./train_debug.sh  > ../../../../perf_log/result_BERT_8.txt 2>&1
cd -
