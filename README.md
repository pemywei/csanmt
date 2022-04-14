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

### 1. The first training phase to optimize the semantic encoder
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

### 2. The second training phase to optimize the NMT model
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup python train.py data-bin/$data_dir
  --distributed-world-size 8-s en -t de \
  --arch transformer_t2t_alitranx \
  --optimizer adam --adam-betas '(0.9, 0.997)' \
  --clip-norm 0.0 \ 
  --encoder-path $encoder_path \
  --share-decoder-input-output-embed \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 1e-07 --warmup-updates 8000 \
  --lr 0.001 --min-lr 1e-09 --weight-decay 0.0 \
  --encoder-lr 0.0 \
  --reset-dataloader \
  --no-progress-bar \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-tokens 4096 \
  --max-epoch 10 \
  --update-freq 2 \ 
  --log-interval 100 \
  --save-interval-updates 1000 \
  --keep-interval-updates 10 \
  --ddp-backend no_c10d \ 
  --seed 1234 \
  --eval-bleu \
  --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
```

### 3. The third training phase to optimize both the NMT model and the semantic encoder
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup python train.py data-bin/$data_dir
  ...
  --encoder-lr 1e-5 \
  ...
```

## Results

| Model                            | En-De | En-Fr|
| -------------------------------- | ----- | -----|
| Transformer (base)               | 27.2  | 38.9 |
| CsaNMT (base)                    | 29.3  | 40.6 |
| -------------------------------- | ----- | -----|
| Transformer (big)                | 28.1  | 40.7 |
| CsaNMT (big)                     | 29.8  | 42.1 |
--[SacreBLEU](https://github.com/mjpost/sacrebleu) Signature: nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0

