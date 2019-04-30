export FLAGS_enable_parallel_graph=0

# Configuration of Allocator and GC
export FLAGS_fraction_of_gpu_memory_to_use=1.0
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_memory_fraction_of_eager_deletion=0.99999

echo "CUDA_VISIBLE_DEVICES: " $CUDA_VISIBLE_DEVICES
echo "FLAGS_enable_parallel_graph: " $FLAGS_enable_parallel_graph
python -c 'import paddle;  print(paddle.__version__)'
python -c 'import paddle;  print(paddle.__git_commit__)'
echo "PYTHONPATH:" $PYTHONPATH

gen_data=/ssd1/transformer_1.1/gen_data
# base model
python -u train.py \
  --src_vocab_fpath $gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --trg_vocab_fpath $gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --train_file_pattern $gen_data/train.tok.clean.bpe.32000.en-de.tiny \
  --token_delimiter ' ' \
  --use_token_batch True \
  --batch_size 4096 \
  --sort_type pool \
  --pool_size 200000 \
  --shuffle False \
  --enable_ce True \
  --shuffle_batch False \
  --use_py_reader True \
  --run_epoch 1 \
  --use_mem_opt True \
  --use_default_pe False \
  --fetch_steps 100  $@ \
  dropout_seed 10 \
  learning_rate 2.0 \
  warmup_steps 8000 \
  beta2 0.997 \
  d_model 512 \
  d_inner_hid 2048 \
  n_head 8 \
  prepostprocess_dropout 0.1 \
  attention_dropout 0.1 \
  relu_dropout 0.1 \
  weight_sharing True \
  pass_num 100 \
  model_dir 'tmp_models' \
  ckpt_dir 'tmp_ckpts'

