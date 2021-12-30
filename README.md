# 인공지능, 훈민정음에 스며들다 2021 한국어 음성·자연어 인공지능 경진대회
**대회 1 대화요약 부문 팀 ICLAB([5위 장려상, 1인참가](https://www.msit.go.kr/bbs/view.do?sCode=user&mId=113&mPid=112&pageIndex=10&bbsSeqNo=94&nttSeqNo=3181143))의 코드 공유 레파지토리입니다.**

***
## Competition Link
http://aihub-competition.or.kr/hangeul
***

## Data
[**한국어 대화 요약 (AI Hub)**](https://aihub.or.kr/aidata/30714)
***

## Structure
```
├── README.md
├── pretrain.py
├── finetune.py
├── rouge_metric.py
├── setup.py
├── vocab_45000.json
├── Custom_Generation.py
```
***
## Docker Image
**Based on nvidia pytorch container 21.10-py3**
```
docker pull kimjonghyeon/hunmin:2.0
```
***

## Requirements
```
CUDA 11.2
pytorch==1.10.0a0+0aef44c
transformers==4.3.3
pycrfsuite_spacing
konlpy
numpy
json
pandas
tqdm
scikit-learn
argparse
```
***
## Hyperparameters
## Model
* Used **Bart-Base** Config

## Pre-Train
* Trained For 10 epochs, removed all dropouts in last epoch<br>
* Only adapted Text Infilling, Random Shuffling is not adapted
* AdamW
* Used Transformers cosine with hard restarts schedule with warmup scheduler 
* Gradient Clipping 
```
lr 1e-4
weight decay None
correct_bias True
```
## Fine-Tune
* Trained For 7 epochs
* AdamW
* Used Transformers cosine schedule with warmup
* Label Smoothing Loss
```
lr 3e-5
weight decay 5e-3
correct_bias False
```

## Generation
```
num_beams 8
repetition_penalty 2.0
no_repeat_ngram_size 3
```
***

## How it works
### 1. Tokenizer
Tokenizer는 Mecab 형태소 분석기를 기반으로 하고, Ai Hub 데이터 기반 45,000개의 Vocab을 사용하였습니다.

<br>

### 2. Pre-Processing
먼저 개인정보 비식별화를 위해 사용된 토큰들(e.g. #@이름#, #@주소#, ...)을 제거해 주었습니다. <br>
그 후 정규표현식을 이용하여 영문자, 숫자, 한글 및 띄어쓰기를 제외한 모든 문자를 제거하였으며, 제거 후 한 글자 발화만 남으면 무의미하다 여겨 이도 함께 제거해 주었습니다.

<br>

### 3. Post-Processing
대회 평가산식이 별도로 공개되지 않았으며, 띄어쓰기에 Score 영향이 있는것으로 보였습니다. <br>
Prediction을 Mecab 형태소 분석 기반 띄어쓰기 시 Rouge Score이 저조하여 CRF (Conditional Random Fields)를 이용한 띄어쓰기 교정을 진행하였습니다.
***
## Tried Techniques
### 1. TextRank
### 2. EDA (Easy Data Augmentation)
### 3. Morpheme-Aware byte-level BPE
***
## References
* [**BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension**](https://arxiv.org/abs/1910.13461)
<br>

* [**TextRank: Bringing Order into Text**](https://aclanthology.org/W04-3252/)

* [**EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks**](https://arxiv.org/abs/1901.11196)
* [**An Empirical Study of Tokenization Strategies
for Various Korean NLP Tasks**](https://arxiv.org/abs/2010.02534)
* [**What Changes Can Large-scale Language Models Bring? Intensive Study on HyperCLOVA: Billions-scale Korean Generative Pretrained Transformers**](https://arxiv.org/abs/2109.04650)

* [**pycrfsuite를 이용한 한국어 띄어쓰기 교정**](https://github.com/lovit/pycrfsuite_spacing)