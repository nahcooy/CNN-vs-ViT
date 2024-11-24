export PYTHONPATH=../../:$PYTHONPATH
CUDA_VISIBLE_DEVICES='0'
python -m torch.distributed.launch --nproc_per_node=2 ../../tools/train_val.py -t
