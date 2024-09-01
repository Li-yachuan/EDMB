# EDMB
Code of paper "EDMB: Edge Detector with Mamba"

## prepare data
As mentioned in paper, following LPCB, RCF and DexiNed in BSDS500, NYUDv2 and BIPED, separately.
## Prepare environment
Version of cuda > 11.6 for Vision Mamba
## Running
### stage I  
```
python main.py --batch_size 3 --stepsize 10-14 --maxepoch 16 --gpu 2 --encoder MIXENC_PNG --decoder MIXUNET --savedir 0806-BSDS-stageII-rand --dataset BSDS-rand --global_ckpt output-VM/0602-bsds-s-randlb/epoch-12-training-record/epoch-12-checkpoint.pth --print_freq 1000
```

