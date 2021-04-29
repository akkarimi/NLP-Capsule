# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import argparse
import random
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F

from tokenization import BertTokenizer
from modeling import BertModel, BertPreTrainedModel, BertLayer, BertPooler
from optimization import BertAdam
from prettytable import PrettyTable
import absa_data_utils as data_utils
from absa_data_utils import ABSATokenizer
import modelconfig
from math import ceil
import matplotlib.pyplot as plt
from layer import PrimaryCaps, FCCaps, FlattenCaps

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


class CapsLayer(nn.Module):
    def __init__(self, config, num_labels):
        super(CapsLayer, self).__init__()
        self.num_labels = num_labels
        self.ngram_size = [2,4,6]
        self.convs_doc = nn.ModuleList([nn.Conv1d(config.max_seq_length, 32, K, stride=2) for K in self.ngram_size])
        torch.nn.init.xavier_uniform_(self.convs_doc[0].weight)
        torch.nn.init.xavier_uniform_(self.convs_doc[1].weight)
        torch.nn.init.xavier_uniform_(self.convs_doc[2].weight)
        self.primary_capsules_doc = PrimaryCaps(num_capsules=config.dim_capsule, in_channels=32, 
                                                out_channels=32, kernel_size=1, stride=1)
        self.flatten_capsules = FlattenCaps()
        self.W_doc = nn.Parameter(torch.FloatTensor(36768, config.num_compressed_capsule))
        torch.nn.init.xavier_uniform_(self.W_doc)
        self.fc_capsules_doc_child = FCCaps(config, 
                                            output_capsule_num=self.num_labels, 
                                            input_capsule_num=config.num_compressed_capsule, 
                                            in_channels=config.dim_capsule, out_channels=config.dim_capsule)
        
    def compression(self, poses, W):
        poses = torch.matmul(poses.permute(0,2,1), W).permute(0,2,1)
        activations = torch.sqrt((poses ** 2).sum(2))
        return poses, activations
        
    def forward(self, embs, labels):
        nets_doc_l = []
        for i in range(len(self.ngram_size)):
            nets = self.convs_doc[i](embs)
            nets_doc_l.append(nets)
        nets_doc = torch.cat((nets_doc_l[0], nets_doc_l[1], nets_doc_l[2]), 2)
        # embs.shape: torch.Size([16, 100, 768])
        # nets_doc_l[0].shape: torch.Size([16, 32, 384])
        # nets_doc.shape: torch.Size([16, 32, 1148])
        poses_doc, activations_doc = self.primary_capsules_doc(nets_doc)
        poses, activations = self.flatten_capsules(poses_doc, activations_doc)
        poses, activations = self.compression(poses, self.W_doc)
        poses, activations = self.fc_capsules_doc_child(poses, activations)
        return poses, activations


class BertForABSA(BertPreTrainedModel):
    def __init__(self, config, num_labels=3):
        super(BertForABSA, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.capsLayer = CapsLayer(config, num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        layers, _, mask = self.bert(input_ids, token_type_ids, 
                                                        attention_mask=attention_mask, 
                                                        output_all_encoded_layers=True)
        poses, activations = self.capsLayer(layers[-1], labels)
        loss = self.loss_fct(activations.view(-1, self.num_labels), labels.view(-1))
        return loss, activations
        
def train(args):

    processor = data_utils.AscProcessor()
    label_list = processor.get_labels()
    tokenizer = ABSATokenizer.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_steps = int(len(train_examples) / args.train_batch_size) * args.num_train_epochs

    train_features = data_utils.convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer, "asc")
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)
    
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    
    train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    
    #>>>>> validation
    if args.do_valid:
        valid_examples = processor.get_dev_examples(args.data_dir)
        valid_features=data_utils.convert_examples_to_features(
            valid_examples, label_list, args.max_seq_length, tokenizer, "asc")
        valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
        valid_all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
        valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
        valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
        valid_data = TensorDataset(valid_all_input_ids, valid_all_segment_ids, valid_all_input_mask, valid_all_label_ids)

        logger.info("***** Running validations *****")
        logger.info("  Num orig examples = %d", len(valid_examples))
        logger.info("  Num split examples = %d", len(valid_features))
        logger.info("  Batch size = %d", args.train_batch_size)

        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.train_batch_size)    

        best_valid_loss=float('inf')
        valid_losses=[]
    #<<<<< end of validation declaration

    model = BertForABSA.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model], num_labels=len(label_list))
    model.cuda()
    
    # Prepare optimizer
    param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad==True]
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)
    
    ########################################################
    # Freeze bert
    for param in model.bert.parameters():
        param.requires_grad = False
    
    # Count number of parameters
    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
    
    count_parameters(model)
    ########################################################
    global_step = 0
    model.train()
    train_loss = []
    validation_loss = []
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.cuda() for t in batch)
            input_ids, segment_ids, input_mask, label_ids = batch
            # input_ids: [batch_size, sequence_length]
            optimizer.zero_grad()
            loss, _ = model(input_ids, segment_ids, input_mask, label_ids)
            loss.backward()
            lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            global_step += 1
        train_loss.append(loss.item())
        print("training loss: ", loss.item(), epoch+1)
        #>>>> perform validation at the end of each epoch.
        new_dirs = os.path.join(args.output_dir, str(epoch+1))
        os.mkdir(new_dirs)
        if args.do_valid:
            model.eval()
            with torch.no_grad():
                losses=[]
                valid_size=0
                for step, batch in enumerate(valid_dataloader):
                    batch = tuple(t.cuda() for t in batch) # multi-gpu does scattering it-self
                    input_ids, segment_ids, input_mask, label_ids = batch
                    loss, _ = model(input_ids, segment_ids, input_mask, label_ids)
                    losses.append(loss.data.item()*input_ids.size(0) )
                    valid_size+=input_ids.size(0)
                valid_loss=sum(losses)/valid_size
                validation_loss.append(valid_loss)
                logger.info("validation loss: %f, epoch: %d", valid_loss, epoch+1)
                valid_losses.append(valid_loss)
                torch.save(model, os.path.join(new_dirs, "model.pt"))
                test(args, new_dirs, dev_as_test=True)
                if epoch == args.num_train_epochs-1:
                    torch.save(model, os.path.join(args.output_dir, "model.pt"))
                    test(args, args.output_dir, dev_as_test=False)
                os.remove(os.path.join(new_dirs, "model.pt"))
            if valid_loss<best_valid_loss:
                best_valid_loss=valid_loss
            model.train()
    if args.do_valid:
        with open(os.path.join(args.output_dir, "valid.json"), "w") as fw:
            json.dump({"valid_losses": valid_losses}, fw)
    else:
        torch.save(model, os.path.join(args.output_dir, "model.pt") )

    plot_train_valid_losses(args.seed, train_loss, validation_loss)

def plot_train_valid_losses(number, train_loss, validation_loss):
    x = range(len(train_loss))
    fig = plt.figure(figsize=(15, 10))
    plt.plot(x, train_loss, label='train loss')
    plt.plot(x, validation_loss, label='valid loss')
    plt.legend()
    plt.savefig('train_valid_loss' + str(number) + '.png', dpi=400)

def test(args, new_dirs=None, dev_as_test=None):  # Load a trained model that you have fine-tuned (we assume evaluate on cpu)    
    processor = data_utils.AscProcessor()
    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])
    if dev_as_test:
        data_dir = os.path.join(args.data_dir, 'dev_as_test')
    else:
        data_dir = args.data_dir
    eval_examples = processor.get_test_examples(data_dir)
    eval_features = data_utils.convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, "asc")

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model = torch.load(os.path.join(new_dirs, "model.pt"))
    model.cuda()
    model.eval()
    
    full_logits=[]
    full_label_ids=[]
    for step, batch in enumerate(eval_dataloader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, segment_ids, input_mask, label_ids = batch
        
        with torch.no_grad():
            _, logits = model(input_ids, segment_ids, input_mask, label_ids)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.cpu().numpy()

        full_logits.extend(logits.tolist())
        full_label_ids.extend(label_ids.tolist())

    output_eval_json = os.path.join(new_dirs, "predictions.json") 
    with open(output_eval_json, "w") as fw:
        json.dump({"logits": full_logits, "label_ids": full_label_ids}, fw)
    
    



def main():    
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", default='bert-base', type=str)

    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir containing json files.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_valid",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    
    parser.add_argument("--num_train_epochs",
                        default=6,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.do_train:
        train(args)
    if args.do_eval:
        test(args)
            
if __name__=="__main__":
    main()