# EDMB
Code of paper "EDMB: Edge Detector with Mamba"

## Prepare data
As mentioned in paper, following LPCB, RCF and DexiNed in BSDS500, NYUDv2 and BIPED, separately.

## Prepare environment
Version of cuda > 11.6 for Vision Mamba

## Download pretrained model
[here](https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_small_0229_ckpt_epoch_222.pth) and put it to dir "model/"

## Training
### Stage I  
```
python main.py --batch_size 4 --stepsize 10-16 --maxepoch 20 --gpu 1 --encoder DUL-Mamba-s --savedir [save dir] --dataset BSDS-rand
```
### Stage II  
```
python main.py --batch_size 3 --stepsize 10-14 --maxepoch 16 --gpu 2 --encoder MIXENC_PNG --decoder MIXUNET --savedir [save dir] --dataset BSDS-rand --global_ckpt [bset result of Stage I]
```
## Test
### Generating multi-granu edge
```
python main.py --batch_size 3 --stepsize 10-14 --maxepoch 16 --gpu 2 --encoder MIXENC_PNG --decoder MIXUNET --savedir [save dir] --dataset BSDS-rand --global_ckpt [bset result of Stage I] --mode test --resume [bset result of Stage II] -mg
```
### Single-granu test
 We used Piotr's Structured Forest matlab toolbox available [here](https://github.com/pdollar/edges).
 
### Multi-granu test
Follow [MuGE](https://github.com/ZhouCX117/UAED_MuGE)

## Acknowledgments
[UAED](https://github.com/ZhouCX117/UAED_MuGE) 
[MuGE](https://github.com/ZhouCX117/UAED_MuGE)
[DiffusionEdge](https://github.com/GuHuangAI/DiffusionEdge)
[VMAMBA](https://github.com/MzeroMiko/VMamba)
