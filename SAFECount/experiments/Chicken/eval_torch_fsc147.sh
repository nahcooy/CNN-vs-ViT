export PYTHONPATH=../../:$PYTHONPATH
CUDA_VISIBLE_DEVICES='0'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 9999 ../../tools/train_val.py -e --config  ./config_fsc147.yaml
