# CsaNMT
PyTorch implementation of CsaNMT: "Learning to Generalize to More: Continuous Semantic Augmentation for Neural Machine Translation" by Xiangpeng Wei.

## Requirements and Installation
- Fairseq v0.10.0
- PyTorch version >= 1.7.0
- python version >= 3.6 

[Link to paper](https://arxiv.org/pdf/)

## Prepare Data
### 1. Get dataset: [WMT14 En-De](https://github.com/pytorch/fairseq/blob/main/examples/translation/prepare-wmt14en2de.sh) and [WMT14 En-Fr](https://github.com/pytorch/fairseq/blob/main/examples/translation/prepare-wmt14en2fr.sh)

### 2. Preprocessed dataset
```bash preprocess_dataset_for_nmt.sh```

## Train
```bash train_alitranx_csanmt.sh```

### 1. The first training to optimize the semantic encoder
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train.py data-bin/$data_dir \
  --distributed-world-size 8 -s en -t de \
  --arch transformer_encoder_alitranx \
  --optimizer adam --adam-betas '(0.98, 0.998)' \
  --clip-norm 0.0 \
  --lr 1e-4
  --max-tokens 4096 \
  --max-epoch 20 \
  --log-interval 100 \
  --save-interval-updates 1000 \
  --keep-interval-updates 10 \
  --seed 1234
```
