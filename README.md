# EDMB
Code of paper "EDMB: Edge Detector with Mamba"

## Prepare data
As mentioned in paper, following LPCB, RCF and DexiNed in BSDS500, NYUDv2 and BIPED, separately.
## Prepare environment
Version of cuda > 11.6 for Vision Mamba
## Running
### Stage I  
```
python main.py --batch_size 4 --stepsize 10-16 --maxepoch 20 --gpu 1 --encoder DUL-Mamba-s --savedir [save dir] --dataset BSDS-rand
```
### Stage II  
```
python main.py --batch_size 3 --stepsize 10-14 --maxepoch 16 --gpu 2 --encoder MIXENC_PNG --decoder MIXUNET --savedir [save dir] --dataset BSDS-rand --global_ckpt [bset result of Stage I]
```
### Multi-granu test
```
python main.py --batch_size 3 --stepsize 10-14 --maxepoch 16 --gpu 2 --encoder MIXENC_PNG --decoder MIXUNET --savedir [save dir] --dataset BSDS-rand --global_ckpt [bset result of Stage I] --mode test --resume [bset result of Stage II] -mg
```
## Test
### Single-granu test
 We used Piotr's Structured Forest matlab toolbox available [here](https://github.com/pdollar/edges).
### Multi-granu test

