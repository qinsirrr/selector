import os
import copy
import json
import os.path
import random
import pickle
from typing import List, Dict, Any
import numpy as np
import torch
import pytorch_lightning as pl
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from urllib.parse import unquote

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _load_json_file(filepath):
    data = []
    if isinstance(filepath, str):
        with open(filepath, 'r', encoding='utf-8') as f:
            d = json.load(f)
            data.extend(d)
    elif isinstance(filepath, list):
        for path in filepath:
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f)
                data.extend(d)
    return data


def _load_pkl_file(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def flatten_results(results: List[Dict]) -> List[Dict]:
    """
    将batch级别的results列表展平为样本级别的列表
    每个batch结果包含多个样本的信息，需要按样本拆分
    """
    flattened = []
    for batch in results:
        # 每个batch的样本数量由rank的长度决定
        batch_size = len(batch['rank'])
        for i in range(batch_size):
            # 为每个样本创建独立的结果字典
            sample_result = {
                'rank': batch['rank'][i],
                'all_rank': batch['all_rank'][i],
                'scores': batch['scores'][i],
                'answer': batch['answers'][i]
            }
            flattened.append(sample_result)
    return flattened

class DataModuleForSelector(pl.LightningDataModule):
    def __init__(self, args):
        super(DataModuleForSelector, self).__init__()
        self.args = args
        self.tokenizer = CLIPProcessor.from_pretrained(self.args.pretrained_model).tokenizer
        self.image_processor = CLIPProcessor.from_pretrained(self.args.pretrained_model).feature_extractor

        # 加载实体知识库
        with open(self.args.data.qid2id, 'r', encoding='utf-8') as f:
            self.qid2id = json.loads(f.readline())
        self.raw_kb_entity = sorted(_load_json_file(self.args.data.entity), key=lambda x: x['id'])
        self.kb_entity = self.setup_dataset_for_entity(self.raw_kb_entity)
        self.kb_id2entity = {raw_ent['id']: ent for raw_ent, ent in zip(self.raw_kb_entity, self.kb_entity)}

        # 加载MIMIC原始数据集（用于Selector的数据划分）
        self.mimic_val_data = self.setup_dataset_for_mention(_load_json_file(self.args.data.dev_file))  # Selector的训练集
        self.mimic_test_data = self.setup_dataset_for_mention(
            _load_json_file(self.args.data.test_file))  # 用于划分Selector的验证集和测试集

        # 加载并展平MIMIC模型的中间结果
        self.validation_results = flatten_results(_load_pkl_file(self.args.data.validation_results))  # 对应MIMIC的验证集
        self.test_results = flatten_results(_load_pkl_file(self.args.data.test_results))

        # 验证数据长度匹配
        assert len(self.mimic_val_data) == len(self.validation_results), \
            f"MIMIC val data length ({len(self.mimic_val_data)}) != validation results length ({len(self.validation_results)})"
        assert len(self.mimic_test_data) == len(self.test_results), \
            f"MIMIC test data length ({len(self.mimic_test_data)}) != test results length ({len(self.test_results)})"

        # 构建Selector的数据集
        self.train_data = self.combine_data_with_results(
            self.mimic_val_data, self.validation_results)

        # 按1:1划分MIMIC测试集作为Selector的验证集和测试集
        split_idx = len(self.mimic_test_data) // 2
        self.val_data = self.combine_data_with_results(
            self.mimic_test_data[:split_idx], self.test_results[:split_idx])
        self.test_data = self.combine_data_with_results(
            self.mimic_test_data[split_idx:], self.test_results[split_idx:])

    def setup_dataset_for_entity(self, data):
        input_data = []
        for sample_dict in tqdm(data, desc='PreProcessing Entities'):
            sample_type = sample_dict['type']
            if sample_type == 'entity':
                entity, attr = unquote(sample_dict.pop('entity_name')), sample_dict.pop('attr')
                input_text = entity + ' [SEP] ' + attr
                input_dict = self.tokenizer(
                    input_text,
                    padding='max_length',
                    max_length=self.args.data.text_max_length,
                    truncation=True
                )
            input_dict['img_list'] = sample_dict['image_list']
            input_dict['sample_type'] = 0 if sample_type == 'entity' else 1
            if 'answer' in sample_dict.keys():
                input_dict['answer'] = self.qid2id[sample_dict['answer']]
            input_data.append(input_dict)
        return input_data

    def setup_dataset_for_mention(self, data):
        input_data = []
        for sample_dict in tqdm(data, desc='PreProcessing Mentions'):
            sample_type = 1
            mention, text = unquote(sample_dict.pop('mentions')), sample_dict.pop('sentence')
            input_text = mention + ' [SEP] ' + text
            input_dict = self.tokenizer(
                input_text,
                padding='max_length',
                max_length=self.args.data.text_max_length,
                truncation=True
            )

            input_dict['img_list'] = [sample_dict['imgPath']] if sample_dict['imgPath'] != '' else []
            input_dict['sample_type'] = sample_type

            # 保留原始答案用于后续验证
            if 'answer' in sample_dict.keys():
                input_dict['answer'] = self.qid2id[sample_dict['answer']]
            if sample_dict.get('answer', '') == 'nil':
                continue
            input_data.append(input_dict)
        return input_data

    def combine_data_with_results(self, data: List[Dict], results: List[Dict]) -> List[Dict]:
        """将原始样本与MIMIC模型输出结果结合，添加预测实体ID和标签"""
        combined = []
        for sample, res in zip(data, results):
            # 从scores中获取预测实体（最大分数对应的实体）
            scores = res['scores']
            pred_ent_idx = scores.argmax()  # 最大分数对应的实体索引
            pred_ent_id = self.raw_kb_entity[pred_ent_idx]['id']  # 映射到实体ID

            # 构造标签：rank=1 → [1,0]，否则→[0,1]
            rank = res['rank']
            label = [1.0, 0.0] if rank == 1 else [0.0, 1.0]

            # 添加到样本中
            sample['pred_ent_id'] = pred_ent_id
            sample['label'] = label
            combined.append(sample)
        return combined

    def choose_image(self, sample_type, img_list, is_eval=False):
        if len(img_list):
            img_name = random.choice(img_list)
            if is_eval:
                img_name = img_list[0]
            if sample_type == 1:
                img_name = img_name.split('/')[-1].split('.')[0] + '.jpg'
            try:
                img_path = os.path.join(
                    self.args.data.kb_img_folder if sample_type == 0 else self.args.data.mention_img_folder,
                    img_name)
                img = Image.open(img_path).resize((224, 224), Image.Resampling.LANCZOS)
                pixel_values = self.image_processor(img, return_tensors='pt')['pixel_values'].squeeze()
            except:
                pixel_values = torch.zeros((3, 224, 224))
        else:
            pixel_values = torch.zeros((3, 224, 224))
        return pixel_values

    def train_collator(self, samples):
        img_list, sample_type, input_dict_list = [], [], []
        pixel_values, pred_ent_ids, labels = [], [], []

        for sample in samples:
            img_list.append(sample.pop('img_list'))
            sample_type.append(sample.pop('sample_type'))
            input_dict_list.append(sample)
            pred_ent_ids.append(sample.pop('pred_ent_id'))
            labels.append(sample.pop('label'))

        # 处理mention的图像
        for idx in range(len(input_dict_list)):
            pixel_values.append(self.choose_image(sample_type[idx], img_list[idx]))

        # 处理mention的文本
        input_dict = self.tokenizer.pad(
            input_dict_list,
            padding='max_length',
            max_length=self.args.data.text_max_length,
            return_tensors='pt'
        )
        input_dict['pixel_values'] = torch.stack(pixel_values)
        input_dict['labels'] = torch.tensor(labels, dtype=torch.float32)

        # 处理预测实体的信息
        ent_info_list = [copy.deepcopy(self.kb_id2entity[idx]) for idx in pred_ent_ids]
        ent_img_list, ent_type, ent_input_dict_list, ent_pixel_values = [], [], [], []
        for ent_dict in ent_info_list:
            ent_img_list.append(ent_dict.pop('img_list'))
            ent_type.append(ent_dict.pop('sample_type'))
            ent_input_dict_list.append(ent_dict)

        # 处理实体的图像
        for idx in range(len(ent_input_dict_list)):
            ent_pixel_values.append(self.choose_image(ent_type[idx], ent_img_list[idx]))

        # 处理实体的文本
        ent_input_dict = self.tokenizer.pad(
            ent_input_dict_list,
            padding='max_length',
            max_length=self.args.data.text_max_length,
            return_tensors='pt'
        )
        ent_input_dict['pixel_values'] = torch.stack(ent_pixel_values)
        ent_input_dict['empty_img_flag'] = torch.tensor(
            [True if not len(_) else False for _ in ent_img_list], dtype=torch.bool
        )

        # 添加实体信息到输入字典
        for k, v in ent_input_dict.items():
            input_dict[f'ent_{k}'] = v

        return input_dict

    def eval_collator(self, samples):
        img_list, sample_type, input_dict_list = [], [], []
        pixel_values, pred_ent_ids, labels = [], [], []

        for sample in samples:
            img_list.append(sample.pop('img_list'))
            sample_type.append(sample.pop('sample_type'))
            input_dict_list.append(sample)
            pred_ent_ids.append(sample.pop('pred_ent_id'))
            labels.append(sample.pop('label'))

        # 处理图像
        for idx in range(len(input_dict_list)):
            pixel_values.append(self.choose_image(sample_type[idx], img_list[idx], is_eval=True))

        # 处理mention文本
        input_dict = self.tokenizer.pad(
            input_dict_list,
            padding='max_length',
            max_length=self.args.data.text_max_length,
            return_tensors='pt'
        )
        input_dict['pixel_values'] = torch.stack(pixel_values)
        # input_dict['answer'] = torch.tensor(gt_ent_ids, dtype=torch.long)
        input_dict['labels'] = torch.tensor(labels, dtype=torch.float32)

         # 处理预测实体的信息
        ent_info_list = [copy.deepcopy(self.kb_id2entity[idx]) for idx in pred_ent_ids]
        ent_img_list, ent_type, ent_input_dict_list, ent_pixel_values = [], [], [], []
        for ent_dict in ent_info_list:
            ent_img_list.append(ent_dict.pop('img_list'))
            ent_type.append(ent_dict.pop('sample_type'))
            ent_input_dict_list.append(ent_dict)
        # 处理实体的图像
        for idx in range(len(ent_input_dict_list)):
            ent_pixel_values.append(self.choose_image(ent_type[idx], ent_img_list[idx]))
        # 处理实体的文本
        ent_input_dict = self.tokenizer.pad(
            ent_input_dict_list,
            padding='max_length',
            max_length=self.args.data.text_max_length,
            return_tensors='pt'
        )
        ent_input_dict['pixel_values'] = torch.stack(ent_pixel_values)
        ent_input_dict['empty_img_flag'] = torch.tensor(
            [True if not len(_) else False for _ in ent_img_list], dtype=torch.bool
        )

        # 添加实体信息到输入字典
        for k, v in ent_input_dict.items():
            input_dict[f'ent_{k}'] = v
        return input_dict

    def entity_dataloader(self):
        return DataLoader(
            self.kb_entity,
            batch_size=self.args.data.embed_update_batch_size,
            num_workers=self.args.data.num_workers,
            shuffle=False,
            collate_fn=self.entity_collator
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.args.data.batch_size,
            num_workers=self.args.data.num_workers,
            shuffle=True,
            collate_fn=self.train_collator
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.args.data.eval_batch_size,
            num_workers=self.args.data.num_workers,
            shuffle=False,
            collate_fn=self.eval_collator
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.args.data.eval_batch_size,
            num_workers=self.args.data.num_workers,
            shuffle=False,
            collate_fn=self.eval_collator
        )

    def entity_collator(self, samples):
        pixel_values, img_list, sample_type, input_dict_list = [], [], [], []
        for sample in samples:
            img_list.append(sample.pop('img_list'))
            sample_type.append(sample.pop('sample_type'))
            input_dict_list.append(sample)

        for idx in range(len(input_dict_list)):
            pixel_values.append(self.choose_image(sample_type[idx], img_list[idx], is_eval=True))

        input_dict = self.tokenizer.pad(
            input_dict_list,
            padding='max_length',
            max_length=self.args.data.text_max_length,
            return_tensors='pt'
        )
        input_dict['pixel_values'] = torch.stack(pixel_values)
        return input_dict