import json
import os
import pickle
import time
import math
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm
from .Selector import SelectorEncoder,Selector

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class LightningForSelector(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.encoder = SelectorEncoder(args)
        self.selector = Selector(args)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        #结果输出表
        self.validation_outputs = []
        self.test_outputs = []
        self.save_dir = os.path.join("./results/model_confidence/selector", self.args.run_name)
        os.makedirs(self.save_dir, exist_ok=True)
    def training_step(self,batch):
        ent_batch = {}
        mention_batch = {}
        num = 1
        #k表示键，v表示值
        for k, v in batch.items():
            # print('k:',k)
            # print('v:',v)
            if k.startswith('ent_'):
                ent_batch[k.replace('ent_', '')] = v
            else:
                mention_batch[k] = v
        # print(batch.items())
        # print(mention_batch)
        # print(ent_batch)
        
        mention_text_embeds, mention_image_embeds=self.encoder(mention_batch['input_ids'],mention_batch['attention_mask'],mention_batch['pixel_values'])
        entity_text_embeds, entity_image_embeds=self.encoder(ent_batch['input_ids'],ent_batch['attention_mask'],ent_batch['pixel_values'])
        combined_embeddings = torch.cat([
            mention_text_embeds,
            mention_image_embeds,
            entity_text_embeds,
            entity_image_embeds
        ], dim=1)  # 在特征维度上拼接
        logits = self.selector(combined_embeddings)
        labels = mention_batch['labels']
        loss = self.loss_fct(logits, labels)
        print(loss)
        return loss
    def validation_step(self, batch, batch_idx):
        # print("validation_batch",batch.keys())
        ent_batch = {}
        mention_batch = {}
        num = 1
        for k, v in batch.items():
            if k.startswith('ent_'):
                ent_batch[k.replace('ent_', '')] = v
            else:
                mention_batch[k] = v


        mention_text_embeds, mention_image_embeds = self.encoder(mention_batch['input_ids'],
                                                                 mention_batch['attention_mask'],
                                                                 mention_batch['pixel_values'])
        entity_text_embeds, entity_image_embeds = self.encoder(ent_batch['input_ids'],
                                                               ent_batch['attention_mask'],
                                                               ent_batch['pixel_values'])
        combined_embeddings = torch.cat([
            mention_text_embeds,
            mention_image_embeds,
            entity_text_embeds,
            entity_image_embeds
        ], dim=1)  # 在特征维度上拼接
        logits = self.selector(combined_embeddings)
        # 只保存logits的第一个值
        first_logits = logits[:, 0].detach().cpu().numpy()
        self.validation_outputs.append(first_logits)
    def on_validation_epoch_end(self):
        # 合并所有batch的结果
        if self.validation_outputs:
            merged_outputs = np.concatenate(self.validation_outputs)
            save_path = os.path.join(self.save_dir, f"validation_confidence.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(merged_outputs, f)
            print(f"验证集合并结果已保存至: {save_path}, 总样本数: {len(merged_outputs)}")
            # 清空验证集输出
            self.validation_outputs = []
    def test_step(self, batch, batch_idx):
        ent_batch = {}
        mention_batch = {}
        num = 1
        for k, v in batch.items():
            if k.startswith('ent_'):
                ent_batch[k.replace('ent_', '')] = v
            else:
                mention_batch[k] = v

        # print(mention_batch)
        # print(ent_batch)
        mention_text_embeds, mention_image_embeds = self.encoder(mention_batch['input_ids'],
                                                                 mention_batch['attention_mask'],
                                                                 mention_batch['pixel_values'])
        entity_text_embeds, entity_image_embeds = self.encoder(ent_batch['input_ids'],
                                                               ent_batch['attention_mask'],
                                                               ent_batch['pixel_values'])
        combined_embeddings = torch.cat([
            mention_text_embeds,
            mention_image_embeds,
            entity_text_embeds,
            entity_image_embeds
        ], dim=1)  # 在特征维度上拼接
        logits = self.selector(combined_embeddings)
        # 只保存logits的第一个值
        first_logits = logits[:, 0].detach().cpu().numpy()
        self.test_outputs.append(first_logits)
    def on_test_epoch_end(self):
        # 合并所有batch的结果
        if self.test_outputs:
            merged_outputs = np.concatenate(self.test_outputs)
            save_path = os.path.join(self.save_dir, f"test_confidence.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(merged_outputs, f)
            print(f"测试集合并结果已保存至: {save_path}, 总样本数: {len(merged_outputs)}")
            # 清空测试集输出
            self.test_outputs = []
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_params = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0001},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_params, lr=self.args.lr, betas=(0.9, 0.999), eps=1e-4)
        return [optimizer]


