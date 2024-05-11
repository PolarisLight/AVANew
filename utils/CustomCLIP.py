import torch
import torchvision
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionTransformer, CLIPEncoderLayer, CLIPEncoder, \
    CLIPVisionEmbeddings
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
import torch.nn.functional as F
import math
from typing import Optional, Union, Tuple

MASK_NUM = 50


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_value_dim, num_heads):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, key_value_dim)
        self.key_proj = nn.Linear(key_value_dim, key_value_dim)
        self.value_proj = nn.Linear(key_value_dim, key_value_dim)
        self.num_heads = num_heads
        self.attention_dropout = nn.Dropout(0.1)
        self.scale = key_value_dim ** -0.5

    def forward(self, query, key, value):
        B, _, D = query.size()
        query = self.query_proj(query).view(B, -1, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        key = self.key_proj(key).view(B, -1, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(B, -1, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention = self.attention_dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attention, value).permute(0, 2, 1, 3).reshape(B, -1, D)
        return context


class CustomCLIPEncoderLayerIn(CLIPEncoderLayer):
    def __init__(self, config):
        super().__init__(config)
        self.cross_attention = CrossAttention(
            config.hidden_size,
            config.hidden_size,
            config.num_attention_heads
        )
        self.layer_norm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
            self,
            hidden_states,
            mask=None,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=False,
            encoder_hidden_states=None,
    ):
        # 原始CLIPEncoderLayer的操作
        layer_outputs = super().forward(
            hidden_states,
            attention_mask,
            causal_attention_mask,
            output_attentions
        )
        hidden_states = layer_outputs[0]

        # 新增的跨注意力层操作
        if self.cross_attention is not None:
            cross_attention_outputs = self.cross_attention(
                mask,
                hidden_states,
                hidden_states
            )
            cross_attention_outputs = self.layer_norm3(cross_attention_outputs)
            # 将跨注意力的输出与自注意力的输出相加
            hidden_states = hidden_states + cross_attention_outputs

        return (hidden_states,) + layer_outputs[1:]


class CustomCLIPEncoder(CLIPEncoder):
    def __init__(self, config):
        super().__init__(config)
        # 使用新的自定义编码器层
        self.layers = nn.ModuleList([CustomCLIPEncoderLayerIn(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            inputs_embeds,
            mask=None,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            encoder_hidden_states=None,  # 新增参数
    ):
        # 保留原始的前向传递代码
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states,
                mask,
                attention_mask,
                causal_attention_mask,
                output_attentions,
                encoder_hidden_states  # 传递给每层
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class CustomCLIPVisionTransformer(CLIPVisionTransformer):
    def __init__(self, config: CLIPVisionConfig, mask_num=MASK_NUM):
        super().__init__(config)
        self.encoder = CustomCLIPEncoder(config)
        self.mask_conv = nn.Conv2d(mask_num, 3, kernel_size=3, padding=1, bias=True)
        self.mask_embedding = CLIPVisionEmbeddings(config)

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
                Returns:

                """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        mask_hidden_states = self.mask_conv(mask)
        mask_hidden_states = self.mask_embedding(mask_hidden_states)
        mask_hidden_states = self.pre_layrnorm(mask_hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            mask=mask_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CustomCLIPVisionModel(CLIPVisionModel):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = CustomCLIPVisionTransformer(config)

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            mask=mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class AesClipCA(nn.Module):
    def __init__(self):
        super(AesClipCA, self).__init__()
        self.clip = CustomCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.proj = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x, mask):
        x = self.clip(x, mask, output_attentions=True)
        self.feature = x.pooler_output
        return self.proj(x.pooler_output)


if __name__ == '__main__':
    vit = AesClipCA()
    print(vit)
    data = torch.randn(2, 3, 224, 224)
    mask = torch.ones(2, 50, 224, 224)
    output = vit(data, mask)
    print(output.shape)