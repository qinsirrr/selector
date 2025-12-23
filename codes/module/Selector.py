import math
import torch
import torch.nn as nn
from transformers import CLIPModel

class SelectorEncoder(nn.Module):
    def __init__(self, args):
        super(SelectorEncoder, self).__init__()
        self.args = args
        self.clip = CLIPModel.from_pretrained(self.args.pretrained_model)
    def forward(self,
                input_ids=None,
                attention_mask=None,
                # token_type_ids=None,
                pixel_values=None):
        clip_output = self.clip(input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values)

        text_embeds = clip_output.text_embeds
        image_embeds = clip_output.image_embeds
        # image_embeds = self.image_cls_fc(image_embeds)
        return text_embeds, image_embeds

class Selector(nn.Module):
    def __init__(self, args):
        super(Selector,self).__init__()
        self.input_size = 512*4
        self.selective_predictor = nn.Sequential(
            nn.Linear(self.input_size, args.selector.hidden_1_dim),
            nn.Dropout(args.selector.dropout),
            nn.ReLU(),
            nn.Linear(args.selector.hidden_1_dim, args.selector.hidden_2_dim),
            nn.Dropout(args.selector.dropout),
            nn.ReLU(),
            nn.Linear(args.selector.hidden_2_dim, args.selector.output_size),
            nn.Softmax(dim=1)
        )
        
        # # 改进的模型架构：添加注意力机制和更复杂的交互
        # self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        # self.layer_norm1 = nn.LayerNorm(512)
        # self.layer_norm2 = nn.LayerNorm(512)
    
        # # 特征融合层
        # self.feature_fusion = nn.Sequential(
        #     nn.Linear(self.input_size, args.selector.hidden_1_dim),
        #     nn.LayerNorm(args.selector.hidden_1_dim),
        #     nn.GELU(),
        #     nn.Dropout(args.selector.dropout),
        # )
        
        # # 置信度预测器
        # self.confidence_predictor = nn.Sequential(
        #     nn.Linear(args.selector.hidden_1_dim, args.selector.hidden_2_dim),
        #     nn.LayerNorm(args.selector.hidden_2_dim),
        #     nn.GELU(),
        #     nn.Dropout(args.selector.dropout),
        #     nn.Linear(args.selector.hidden_2_dim, args.selector.hidden_2_dim // 2),
        #     nn.LayerNorm(args.selector.hidden_2_dim // 2),
        #     nn.GELU(),
        #     nn.Dropout(args.selector.dropout),
        #     nn.Linear(args.selector.hidden_2_dim // 2, args.selector.output_size)
        # )
        
    def forward(self, embeddings):
        # batch_size = embeddings.size(0)
        
        # # 重塑为序列格式用于注意力机制
        # # embeddings: [batch_size, 2048] -> [batch_size, 4, 512]
        # seq_embeddings = embeddings.view(batch_size, 4, 512)
        
        # # 自注意力机制
        # attn_output, _ = self.attention(seq_embeddings, seq_embeddings, seq_embeddings)
        # attn_output = self.layer_norm1(seq_embeddings + attn_output)
        
        # # 前馈网络
        # ff_output = self.layer_norm2(attn_output)
        
        # # 重新展平
        # fused_embeddings = ff_output.view(batch_size, -1)
        
        # # 特征融合
        # fused_features = self.feature_fusion(fused_embeddings)
        
        # # 置信度预测
        # logits = self.confidence_predictor(fused_features)
        logits = self.selective_predictor(embeddings)
        return logits