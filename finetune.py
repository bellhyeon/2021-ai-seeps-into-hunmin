# NSML
import nsml
from nsml import HAS_DATASET, DATASET_PATH, IS_ON_NSML

# Commom
import numpy as np
import json
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split
import random
import argparse
from rouge_metric import Rouge
import time
from typing import List
import shutil

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# transformers
from transformers.models.bart.modeling_bart import BartConfig
from Custom_Generation import BartForConditionalGeneration
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

# tokenizer
from konlpy.tag import Mecab

# pycrfsuite spacing
from pycrfsuite_spacing import TemplateGenerator
from pycrfsuite_spacing import CharacterFeatureTransformer
from pycrfsuite_spacing import PyCRFSuiteSpacing

class Preprocess:
    def __init__(self):
        super(Preprocess, self).__init__()

    @staticmethod
    def train_data_loader(root_path):
        print('Load Train Path...')
        train_path = os.path.join(root_path, 'train', 'train_data', '*')
        pathes = glob(train_path)
        return pathes

    @staticmethod
    def test_data_loader(root_path):
        test_path = os.path.join(root_path, 'test', 'test_data', '*')
        pathes = glob(test_path)
        return pathes

    @staticmethod
    def make_dataset_list(path_list):
        print('Load Json Data...')
        json_data_list = []

        for path in path_list:
            with open(path) as f:
                json_data_list.append(json.load(f))

        return json_data_list

    @staticmethod
    def make_set_as_df(dataset, is_train=True):
        print('Make Train DataFrame...')
        if is_train:
            train_dialogue = []
            train_dialogue_id = []
            train_summary = []
            for topic in dataset:
                for data in tqdm(topic['data']):
                    train_dialogue_id.append(data['header']['dialogueInfo']['dialogueID'])
                    train_dialogue.append(([sub.remove_masking1(sub.remove_masking(dialogue['utterance']))
                                                                for dialogue in data['body']['dialogue']]))
                    train_summary.append(data['body']['summary'])

            train_data = pd.DataFrame(
                {
                    'dialogueID': train_dialogue_id,
                    'dialogue': train_dialogue,
                    'summary': train_summary,
                }
            )
            return train_data

        else:
            test_dialogue = []
            test_dialogue_id = []
            for topic in dataset:
                for data in tqdm(topic['data']):
                    test_dialogue_id.append(data['header']['dialogueInfo']['dialogueID'])
                    test_dialogue.append(([sub.remove_masking1(sub.remove_masking(dialogue['utterance']))
                                                                for dialogue in data['body']['dialogue']]))

            test_data = pd.DataFrame(
                {
                    'dialogueID': test_dialogue_id,
                    'dialogue': test_dialogue
                }
            )
            return test_data

    @staticmethod
    # 영문자, 숫자, 한글 및 띄어쓰기를 제외한 모든 특수문자 제거, 의미가 없다고 여긴 한 글자 발화는 모두 제거.
    def remove_special_char(dialogues: List):
        cleaned_dialogues = []
        regx = re.compile(r"[^ a-zA-Z0-9가-힣]")
        for dialogue in dialogues:
            cleaned_sent = []
            for sent in dialogue:
                cleaned = regx.sub(" ", sent)
                cleaned = ' '.join(cleaned.split())
                if not cleaned == '' and len(cleaned) != 1:
                    cleaned_sent.append(cleaned)
            cleaned_dialogues.append(cleaned_sent)
        return cleaned_dialogues

class RougeScorer:
    def __init__(self):
        self.rouge_evaluator = Rouge(
            metrics=["rouge-n", "rouge-l"],
            max_n=2,
            limit_length=True,
            length_limit=1000,
            length_limit_type="words",
            use_tokenizer=True,
            apply_avg=True,
            apply_best=False,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
        )

    def compute_rouge(self, ref, hyp):
        scores = self.rouge_evaluator.get_scores(ref, hyp)
        str_scores = self.format_rouge_scores(scores)
        return str_scores

    def format_rouge_scores(self, scores):
    	return "rouge-1: {:.8f}, rouge-2: {:.8f}, rouge-l: {:.8f}".format(
            scores["rouge-1"]["f"],
            scores["rouge-2"]["f"],
            scores["rouge-l"]["f"],
        )

class sub:
    '''
    Special Masking 제거
    ex)
    #@이모티콘# -> # -> ''
    #@이모티콘#쑥스# -> #쑥스# -> ''
    '''
    @staticmethod
    def remove_masking(sentence):
        sentence = re.sub(r'#(.+?)#', '#', sentence)
        return sentence

    @staticmethod
    def remove_masking1(sentence):
        sentence = re.sub(r'#(.+?)#', '', sentence)
        return sentence

def tokenize(word_to_idx, morphs):
    tokenized = []
    for m in morphs:
        if word_to_idx.get(m):
            tokenized.append(word_to_idx[m])
        else:
            tokenized.append(2)    # <unk>
    return tokenized

def batch_decode(idx_to_word, tokens):
    output = []
    for token in tokens:
        sentence = ""
        for t in token:
            if idx_to_word.get(t):
                if t == 0 or t == 2 or t == 3 or t == 4:    # <s>, <unk>, <pad>, <mask>
                    pass
                elif t == 1:    # </s>
                    break
                else:
                    sentence += idx_to_word[t]
            else:
                sentence += ""
            sentence += " "
        output.append(sentence)
    return output

class BARTSummaryDataset(Dataset):
    def __init__(self, df, tokenizer, word_to_idx, max_len, pad_index=None, ignore_index=-100, train=True):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.word_to_idx = word_to_idx
        if pad_index is None:
            self.pad_index = 3
        else:
            self.pad_index = pad_index
        self.ignore_index = ignore_index
        self.train = train
        self.eos_token_id = 1

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return torch.tensor(inputs)

    def add_ignored_data(self, inputs):
        '''
        Add -100 token to Label For pytorch CrossEntropyLoss
        '''
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return torch.tensor(inputs)

    def __getitem__(self, idx):
        '''
        input_id: <s> dialogue </s>
        label_id: <s> label </s>
        dec input_ids: </s><s> label -> label id shifting
        '''
        
        instance = self.df.iloc[idx]
        dialogue_morphs = self.tokenizer.morphs(instance['dialogue'])
        input_ids = [0] + tokenize(self.word_to_idx, dialogue_morphs)   # <s> + dialogue
        input_ids.append(self.eos_token_id)    # <s> + dialogue + </s>
        input_ids = self.add_padding_data(input_ids)  # <s> + dialogue + </s> + <pad>

        if self.train:
            label_morphs = self.tokenizer.morphs(instance['summary'])
            label_ids = [0] + tokenize(self.word_to_idx, label_morphs)    # <s> + label
            label_ids.append(self.eos_token_id)    # <s> + label + </s>

            dec_input_ids = [1] # </s>
            dec_input_ids += label_ids[:-1] # </s><s> + label   -> Label Shifting
            dec_input_ids = self.add_padding_data(dec_input_ids)    # </s><s> + label + <pad>

            label_ids = self.add_ignored_data(label_ids)    # <s> + label + </s> + [-100]

            return {'input_ids': np.array(input_ids, dtype=np.int_),
                    'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                    'labels': np.array(label_ids, dtype=np.int_),
                    'summary': instance['summary']
                   }

        else:
            return {'input_ids': np.array(input_ids, dtype=np.int_),
                    }

    def __len__(self):
        return len(self.df)

'''
Init Weights For from_pretrained

def init_weights(model):
    for n, p in model.named_parameters():
        if 'weight' in n:
            p.data.normal_(mean=0.0, std=2e-2)
        if 'bias' in n:
            p.data.zero_()
'''

### BART ####
class Bart(nn.Module):
    def __init__(self, config):
        super(Bart, self).__init__()
        self.model = BartForConditionalGeneration(config=config)
        self.model.init_weights()    # initialize Model Weights
        # init_weights(self.model)    # init weights
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.pad_token_id = 3

    def forward(self, inputs):
        input_ids = inputs['input_ids'].cuda()
        labels = inputs['labels'].cuda()
        decoder_input_ids = inputs['decoder_input_ids'].cuda()

        attention_mask = input_ids.ne(self.pad_token_id).float()
        decoder_attention_mask = decoder_input_ids.ne(self.pad_token_id).float()

        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          decoder_attention_mask=decoder_attention_mask,
                          decoder_input_ids=decoder_input_ids,
                          labels=labels,
                          return_dict=True)

    def generate(self, inputs):
        attention_mask = inputs.ne(self.pad_token_id).float()
        return self.model.generate(
                    inputs,
                    num_beams=8,
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=3,
                    eos_token_id=1,
                    num_return_sequences=1,
                    min_length=5,
                    max_length=75,
                    pad_token_id=3,
                    bos_token_id=0,
                    decoder_start_token_id=1,
                    attention_mask=attention_mask
        )

def train_spacing(dialogues_to_train):

    to_feature = CharacterFeatureTransformer(
            TemplateGenerator(begin=-2,
                              end=2,
                              min_range_length=3,
                              max_range_length=3)
        )

    spacing = PyCRFSuiteSpacing(
        to_feature=to_feature,
        feature_minfreq=3,  # default = 0
        max_iterations=100,
        l1_cost=1.0,
        l2_cost=1.0,
        verbose=True
    )

    spacing.train(dialogues_to_train, 'spacing.crfsuite')

    return spacing


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    print(f'Global seed set to {seed}')
    return seed

def train(epoch, train_dataloader, optimizer, scheduler):
    scaler = torch.cuda.amp.GradScaler()
    total_train_loss = []
    train_time = time.time()
    print(f'******EPOCH {epoch + 1}******')
    model.train()
    for batch_item in train_dataloader:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(batch_item)
            loss = outputs['loss']
            total_train_loss.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    print(f'Epoch: {epoch + 1}, train_loss: {np.mean(total_train_loss)}')
    print(f'train took {time.time() - train_time}s')

def validation(valid_dataloader, model):
    # Validation
    model.eval()
    pred_list = []
    sample_list = []
    total_val_loss = []
    val_start_time = time.time()
    with torch.no_grad():
        for batch_item in valid_dataloader:
            outputs = model(batch_item)
            loss = outputs['loss']
            pred = model.generate(batch_item['input_ids'].cuda())
            pred = batch_decode(idx_to_word, pred.cpu().numpy())
            pred_temp = []
            for p in pred:
                pred_temp.append(spacing.correct(''.join(p.split())))
            total_val_loss.append(loss.item())
            pred_list.extend(pred_temp)
            sample_list.extend(batch_item['summary'])
    print(f'Epoch: {epoch + 1}, valid_loss: {np.mean(total_val_loss)}')
    print(f'Epoch: {epoch + 1}, {rouge.compute_rouge(pred_list, sample_list)}')
    print(f'pred: {pred_list[:5]}')
    print(f'answer: {sample_list[:5]}')
    print(f'validation took {time.time() - val_start_time}s')

def bind_model(parser):
    # 학습한 모델을 저장하는 함수입니다.

    def save(dir_name, *parser):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        save_dir = os.path.join(dir_name, 'model.pt')
        spacing_save_dir = os.path.join(dir_name, 'spacing.crfsuite')
        shutil.copyfile('spacing.crfsuite', spacing_save_dir)
        checkpoint = {"model": model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict()}
        torch.save(checkpoint, save_dir)
        word_save_dir = os.path.join(dir_name, 'vocab_45000.json')
        shutil.copyfile('vocab_45000.json', word_save_dir)
        print("저장 완료!")

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *parser):
        model_save_dir = os.path.join(dir_name, 'model.pt')
        checkpoint = torch.load(model_save_dir, map_location='cuda')
        model.load_state_dict(checkpoint['model'], strict=False)

        global word_to_idx
        with open(os.path.join(dir_name, 'vocab_45000.json'), 'r', encoding='utf-8') as f:
            word_to_idx = json.load(f)

        global spacing
        to_feature = CharacterFeatureTransformer(
            TemplateGenerator(begin=-2,
                              end=2,
                              min_range_length=3,
                              max_range_length=3)
        )

        spacing = PyCRFSuiteSpacing(
            to_feature=to_feature,
            feature_minfreq=3,  # default = 0
            max_iterations=100,
            l1_cost=1.0,
            l2_cost=1.0
        )

        spacing.load_tagger(os.path.join(dir_name, 'spacing.crfsuite'))

        print("로딩 완료!")

    def infer(test_path, **kwparser):
        tokenizer = Mecab()
        model.cuda()
        model.eval()
        preprocessor = Preprocess()
        result_list = []

        test_path_list = preprocessor.test_data_loader(DATASET_PATH)
        test_path_list.sort()
        print(f'test_path_list :\n{test_path_list}')

        test_json_list = preprocessor.make_dataset_list(test_path_list)
        test_data = preprocessor.make_set_as_df(test_json_list, is_train=False)
        print(len(test_data))
        dialogue = preprocessor.remove_special_char(test_data['dialogue'].tolist())

        test_data['dialogue'] = [' '.join(x) for x in dialogue]
        print(test_data.head())
        print(f'test_data:\n{test_data["dialogue"]}')

        idx_to_word = dict(zip(word_to_idx.values(), word_to_idx.keys()))

        test_dataset = BARTSummaryDataset(df=test_data, train=False, tokenizer=tokenizer, word_to_idx=word_to_idx,
                                             max_len=200)
        test_data_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)

        with torch.no_grad():
            for data in tqdm(test_data_loader):
                input_ids = data['input_ids'].cuda()
                generated = model.generate(input_ids)
                generated = generated.cpu().numpy()
                output = batch_decode(idx_to_word, generated)
                output_temp = []
                for o in output:
                    output_temp.append(spacing.correct(''.join(o.split())))
                print(output_temp)
                result_list.extend(output_temp)
        test_id = test_data['dialogueID']

        # DONOTCHANGE: They are reserved for nsml
        return list(zip(test_id, result_list))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BART')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('warmup_steps', type=int, default=10000)
    parser.add_argument('seed', type=int, default=42)
    args = parser.parse_args()
    bind_model(parser=args)

    # Resize Model Size (Follows Bart-Base Config)
    config = BartConfig(
        vocab_size=45000,
        activation_dropout=0.1,
        attention_dropout=0.1,
        classifier_dropout=0.0,
        d_model=768,
        encoder_layers=6,
        decoder_layers=6,
        encoder_attention_heads=12,
        decoder_attention_heads=12,
        encoder_ffn_dim=3072,
        decoder_ffn_dim=3072,
        decoder_start_token_id=1,
        pad_token_id=3,
        eos_token_id=1,
        max_position_embeddings=200
    )

    model = Bart(config)

    if args.pause:
        nsml.paused(scope=locals())

    if args.mode == 'train':
        nsml.load(checkpoint='0', session='final_dialogue/377')
        preprocessor = Preprocess()

        seed_everything(args.seed)
        torch.set_num_threads(4)

        train_path_list = preprocessor.train_data_loader(DATASET_PATH)
        train_path_list.sort()

        train_json_list = preprocessor.make_dataset_list(train_path_list)

        train_data = preprocessor.make_set_as_df(train_json_list)

        dialogue = train_data['dialogue'].tolist()
        summary = train_data['summary'].tolist()
        cleaned_dialogues = preprocessor.remove_special_char(dialogue)  # 특수문자 제거된 dialogues
        concated_dialogues = [' '.join(x) for x in cleaned_dialogues]  # join
        train_data['dialogue'] = concated_dialogues

        temp = concated_dialogues
        temp.extend(summary)

        train_spacing(temp)    # train spacing pattern

        print(train_data.tail())

        print(f'Number of Train Data:{len(train_data)}')

        with open('vocab_45000.json', 'r', encoding='utf-8') as f:
            word_to_idx = json.load(f)

        idx_to_word = dict(zip(word_to_idx.values(), word_to_idx.keys()))

        tokenizer = Mecab()

        max_length = args.max_length  # Encoder Max Length
        batch_size = args.batch_size
        epochs = args.epochs
        learning_rate = args.lr
        device = torch.device("cuda:0")

        train_df, val_df = train_test_split(train_data, test_size=0.1)

        train_dataset = BARTSummaryDataset(df=train_df, tokenizer=tokenizer, word_to_idx=word_to_idx,
                                             max_len=max_length)
        valid_dataset = BARTSummaryDataset(df=val_df, tokenizer=tokenizer, word_to_idx=word_to_idx,
                                             max_len=max_length)

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True, num_workers=args.num_workers)

        valid_dataloader = DataLoader(valid_dataset,
                                      batch_size=batch_size,
                                      shuffle=False, num_workers=args.num_workers)
        model = model.to(device)
        rouge = RougeScorer()

        num_train_steps = len(train_dataloader) * epochs
        print(f'total train steps: {num_train_steps}')

        optimizer = AdamW(model.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay, correct_bias=False)  # Weight Decay Prevents Overfitting

        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_steps)

        for epoch in range(epochs):
            train(epoch=epoch, train_dataloader=train_dataloader, optimizer=optimizer, scheduler=scheduler)
            validation(valid_dataloader=valid_dataloader, model=model)

            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)