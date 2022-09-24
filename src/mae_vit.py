# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np
import mindspore as ms

from mindspore import nn
from mindspore import Tensor
from mindspore import ops as P
from mindspore import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.nn.transformer import TransformerEncoder, TransformerDecoder
from mindspore.common.initializer import initializer, XavierUniform

from utils import load_pretrained
from registry import register_model
from mission.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT

'''
1. 添加了Patchify类与PatchEmbed类;
2. 添加了get_2d_sincos_pos_embed位置编码API以及其下层API;
3. 添加了ImageNet数据集的预处理配置_cfg(RandomColorAdjust与HWC2CHW定义在公用接口transforms_factory.py中)，参数设定与
   pengzhiliang的torch实现稍有不同;
4. github中stars较高的facebook的torch实现，将encoder与decoder统一定义在MaskedAutoencoderViT类中，而pengzhiliang的torch实现，
   将Encoder与Decoder分别定义为类，再在PretrainVisionTransformer中实例化，个人认为这两种定义方式结构更加清晰;
5. 原实现中没有模型注册机制，参照pengzhiliang的torch实现加入了pretrain_mae_small_patch16_224、pretrain_mae_base_patch16_224、
   pretrain_mae_large_patch16_224三种规模的模型(encoder与decoder相关参数按比例缩放);
6. 暂时未添加网络的测试main()函数;
'''


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000,
        'dataset_transform': {
            'transforms_imagenet_train': {
                'image_resize': 224,
                'hflip': 0.5,
                'interpolation': 'bicubic',
                'mean': IMAGENET_DEFAULT_MEAN / 255,
                'std': IMAGENET_DEFAULT_STD / 255,
            },
            'transforms_imagenet_eval': {
                # int(256 / 224 * image_size)
                'image_resize': int(256 / 224 * 224),
                'crop_pct': DEFAULT_CROP_PCT,
                'interpolation': 'bicubic',
                'mean': IMAGENET_DEFAULT_MEAN,
                'std': IMAGENET_DEFAULT_STD,
            },
        },
        **kwargs
    }


default_cfgs = {
    'pretrain_mae_small_patch16_224': _cfg(url=''),
    'pretrain_mae_base_patch16_224': _cfg(url=''),
    'pretrain_mae_large_patch16_224': _cfg(url='')
}


class Patchify(nn.Cell):
    """Patchify origin image to patches."""

    def __init__(self, patch_size):
        super(Patchify, self).__init__()

        self.patch_size = patch_size
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, img):
        p = self.patch_size
        bs, channels, h, w = img.shape
        x = self.reshape(img, (bs, channels, h // p, p, w // p, p))
        x = self.transpose(x, (0, 2, 4, 1, 3, 5))
        patches = self.reshape(x, (bs, (h // p) * (w // p), channels * p * p))
        return patches


class PatchEmbed(nn.Cell):
    def __init__(self, img_size=224, patch_size=16, in_features=3, out_features=768):
        super(PatchEmbed, self).__init__()
        self.hybrid = None
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.projection = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                                    kernel_size=patch_size, stride=patch_size, has_bias=True)
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.norm = nn.LayerNorm((out_features,), epsilon=1e-6).to_float(ms.float32)

    def construct(self, x):
        x = self.projection(x)
        x = self.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x = self.transpose(x, (0, 2, 1))
        x = self.norm(x)
        return x


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# MoCo v3: https://github.com/facebookresearch/moco-v3

# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)

    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    # 按一维堆叠为(196, 1024)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class MAEModule(nn.Cell):
    """Base Module For MAE."""

    def __init__(self, batch_size, image_size, patch_size, masking_ratio=0.75, channels=3):
        super(MAEModule, self).__init__()
        assert 0 < masking_ratio < 1, \
            'masking ratio must be kept between 0 and 1'
        # seq_length
        self.num_patches = (image_size // patch_size) ** 2
        # seq masked number
        self.num_masked = int(masking_ratio * self.num_patches)
        # batch range
        self.batch_range = np.arange(batch_size)[:, None]
        # per patch dim
        self.patch_dim = channels * patch_size ** 2
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        self.rand_indices = P.Sort()(P.UniformReal()((batch_size, self.num_patches)))
        self.masked_indices = self.rand_indices[1][:, :self.num_masked]
        self.unmasked_indices = self.rand_indices[1][:, self.num_masked:]
        self.mask_info = None
        self.encoder = None
        self.decoder = None

    def generate_mask(self):
        self.mask_info = {
            "batch_range": self.batch_range,
            "masked_indices": self.masked_indices,
            "unmasked_indices": self.unmasked_indices,
        }

        return self.mask_info

    def encoder_engine(self):
        """tokens encoder."""
        return self.encoder

    def decoder_engine(self):
        """code decoder."""
        return self.decoder


class MaeEncoder(MAEModule):
    """MAE Encoder, Default is Vit."""

    def __init__(self,
                 batch_size,
                 patch_size,
                 image_size,
                 encoder_layers=12,
                 encoder_num_heads=12,
                 encoder_dim=768,
                 mlp_ratio=4,
                 masking_ratio=0.75,
                 drop_path=0.1,
                 channels=3,
                 initialization=XavierUniform()):
        super(MaeEncoder, self).__init__(batch_size, image_size, patch_size, masking_ratio, channels)

        self.seq_length = self.num_patches - self.num_masked + 1
        self.encoder = TransformerEncoder(batch_size=batch_size, num_layers=encoder_layers,
                                          num_heads=encoder_num_heads, hidden_size=encoder_dim,
                                          ffn_hidden_size=encoder_dim * mlp_ratio,
                                          seq_length=self.seq_length, hidden_dropout_rate=drop_path)
        cls_token = Parameter(
            initializer(initialization, (1, 1, encoder_dim)),
            name='cls', requires_grad=True
        )
        self.cls_token = P.Tile()(cls_token, (batch_size, 1, 1))

        self.encoder_pos_embedding = Parameter(
            initializer(initialization, (1, self.num_patches + 1, encoder_dim)),
            name='pos_embedding', requires_grad=False
        )
        self.add = P.Add()
        self.cat = P.Concat(axis=1)
        self.stride_slice = P.StridedSlice()
        self.norm = nn.LayerNorm((encoder_dim,), epsilon=1e-6).to_float(mstype.float32)
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size,
                                      in_features=channels, out_features=encoder_dim)

        self.encoder_input_mask = Tensor(np.ones((batch_size, self.seq_length, self.seq_length)), mstype.float32)

        self._init_weights()

    def _init_weights(self):
        encoder_pos_emd = Tensor(
            get_2d_sincos_pos_embed(self.encoder_pos_embedding.shape[-1],
                                    int(self.num_patches ** .5),
                                    cls_token=True),
            mstype.float32
        )
        self.encoder_pos_embedding.set_data(P.ExpandDims()(encoder_pos_emd, 0))

    def construct(self, img):
        # patch to encoder tokens and add positions
        tokens = self.patch_embed(img)
        encoder_pos_embedding = self.stride_slice(self.encoder_pos_embedding, (0, 1, 0),
                                                  (1, self.encoder_pos_embedding.shape[1],
                                                   self.encoder_pos_embedding.shape[2]), (1, 1, 1))
        tokens = self.add(tokens, encoder_pos_embedding)
        # get the unmasked tokens to be encoded
        tokens = tokens[self.batch_range, self.unmasked_indices]

        # cls_tokens add pos_embedding
        cls_pos_embedding = self.stride_slice(self.encoder_pos_embedding, (0, 0, 0),
                                              (1, 1, self.encoder_pos_embedding.shape[2]),
                                              (1, 1, 1))
        # cls_tokens = self.add(self.cls_token, self.encoder_pos_embedding[:, :1, :])
        cls_tokens = self.add(self.cls_token, cls_pos_embedding)

        # concat cls_tokens
        tokens = self.cat((cls_tokens, tokens))

        # attend with vision transformer
        encoded_tokens = self.encoder(tokens, self.encoder_input_mask)[0]
        encoded_tokens = self.norm(encoded_tokens)

        return encoded_tokens


class PreTrainMAEVit(MAEModule):
    """Pretrain MAEVit Module."""

    def __init__(self,
                 batch_size,
                 patch_size,
                 image_size,
                 encoder_layers=12,
                 encoder_num_heads=12,
                 encoder_dim=768,
                 decoder_layers=8,
                 decoder_num_heads=16,
                 decoder_dim=512,
                 mlp_ratio=4,
                 masking_ratio=0.75,
                 drop_path=0.1,
                 channels=3,
                 norm_pixel_loss=False,
                 initialization=XavierUniform()):
        super(PreTrainMAEVit, self).__init__(batch_size, image_size, patch_size, masking_ratio, channels)
        self.encoder = MaeEncoder(batch_size, patch_size, image_size,
                                  encoder_layers=encoder_layers,
                                  encoder_dim=encoder_dim,
                                  encoder_num_heads=encoder_num_heads,
                                  mlp_ratio=mlp_ratio,
                                  drop_path=drop_path,
                                  initialization=initialization)
        # decoder parameters
        self.seq_length = self.encoder.seq_length
        tgt_seq_length = self.num_patches + 1
        self.mask_token = Parameter(P.StandardNormal()((decoder_dim,)))
        self.mask_tokens = P.Tile()(self.mask_token, (batch_size, self.num_masked, 1))
        self.enc_to_dec = nn.Dense(encoder_dim, decoder_dim,
                                   has_bias=True) if encoder_dim != decoder_dim else P.Identity()
        self.decoder = TransformerDecoder(batch_size=batch_size,
                                          num_layers=decoder_layers,
                                          num_heads=decoder_num_heads,
                                          hidden_size=decoder_dim,
                                          ffn_hidden_size=decoder_dim * mlp_ratio,
                                          src_seq_length=self.seq_length,
                                          tgt_seq_length=tgt_seq_length)
        decoder_pos_emd = Tensor(
            get_2d_sincos_pos_embed(decoder_dim, int(self.num_patches ** .5),
                                    cls_token=True), mstype.float32
        )
        self.decoder_pos_embedding = nn.Embedding(tgt_seq_length, decoder_dim, embedding_table=decoder_pos_emd)
        self.decoder_pos_embedding.requires_grad = False
        self.attention_mask = Tensor(np.ones((batch_size, tgt_seq_length, tgt_seq_length)), mstype.float32)

        self.to_pixels = nn.Dense(decoder_dim, self.patch_dim, has_bias=True)
        self.decoder_norm = nn.LayerNorm((decoder_dim,), epsilon=1e-6).to_float(mstype.float32)

        self.patchify = Patchify(patch_size=patch_size)

        self.add = P.Add()
        self.divide = P.Div()
        self.cast = P.Cast()
        self.cat = P.Concat(axis=1)
        self.pow = P.Pow()
        self.mean = P.ReduceMean(keep_dims=True)
        self.norm_pixel_loss = norm_pixel_loss
        self.mse_loss = nn.MSELoss()

    def calc_loss(self, pred, target):
        pred = self.cast(pred, mstype.float32)
        target = self.cast(target, mstype.float32)
        if self.norm_pixel_loss:
            mean = self.mean(target, -1)
            var = target.var(axis=-1, keepdims=True)
            target = self.divide(target - mean, self.pow(var + 1e-6, 0.5))
        recon_loss = self.mse_loss(pred, target)
        return recon_loss

    def construct(self, img, label=None):
        # tokens encoder
        encoder_tokens = self.encoder(img)
        patches = self.patchify(img)

        # project encoder to decoder dimensions,
        # if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoder_tokens)

        # add position embedding for decoder tokens
        img_tokens = decoder_tokens[:, 1:, :]
        cls_tokens = decoder_tokens[:, :1, :]
        decoder_tokens_ = self.add(img_tokens, self.decoder_pos_embedding(self.unmasked_indices))
        decoder_tokens = self.cat((cls_tokens, decoder_tokens_))

        # mask tokens add the positions using the masked indices derived above
        mask_tokens = self.add(self.mask_tokens, self.decoder_pos_embedding(self.masked_indices))

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = self.cat((decoder_tokens, mask_tokens))
        decoded_tokens = self.decoder(decoder_tokens, self.attention_mask)[0]

        # normalize decoder tokens
        decoded_tokens = self.decoder_norm(decoded_tokens)

        # project to pixel values for whole tokens
        decoded_tokens = decoded_tokens[:, 1:, :]
        pred_pixel_values = self.to_pixels(decoded_tokens)

        # sorted patches according to indices
        masked_patches = patches[self.batch_range, self.masked_indices]
        unmasked_patches = patches[self.batch_range, self.unmasked_indices]
        sort_patches = self.cat((unmasked_patches, masked_patches))
        # calculate reconstruction loss
        loss = self.calc_loss(pred_pixel_values, sort_patches)
        return loss


@register_model
def pretrain_mae_small_patch16_224(pretrained: bool = False,
                                   num_classes: int = 1000,
                                   channels: int = 3,
                                   **kwargs):
    model_args = default_cfgs['pretrain_mae_small_patch16_224']

    model = PreTrainMAEVit(
        batch_size=8,
        patch_size=16,
        image_size=224,
        encoder_layers=12,
        encoder_num_heads=6,
        encoder_dim=384,
        decoder_layers=8,
        decoder_num_heads=8,
        decoder_dim=256,
        mlp_ratio=4,
        masking_ratio=0.75,
        drop_path=0.1,
        initialization=XavierUniform()
        ** kwargs)

    model.dataset_transform = model_args['dataset_transform']

    if pretrained:
        load_pretrained(model, model_args, num_classes=num_classes, in_channels=channels)

    return model


@register_model
def pretrain_mae_base_patch16_224(pretrained: bool = False,
                                  num_classes: int = 1000,
                                  channels: int = 3,
                                  **kwargs):
    model_args = default_cfgs['pretrain_mae_base_patch16_224']

    model = PreTrainMAEVit(
        batch_size=8,
        patch_size=16,
        image_size=224,
        encoder_layers=12,
        encoder_num_heads=12,
        encoder_dim=768,
        decoder_layers=8,
        decoder_num_heads=16,
        decoder_dim=512,
        mlp_ratio=4,
        masking_ratio=0.75,
        drop_path=0.1,
        initialization=XavierUniform()
        ** kwargs)

    model.dataset_transform = model_args['dataset_transform']

    if pretrained:
        load_pretrained(model, model_args, num_classes=num_classes, in_channels=channels)

    return model


@register_model
def pretrain_mae_large_patch16_224(pretrained: bool = False,
                                   num_classes: int = 1000,
                                   channels: int = 3,
                                   **kwargs):
    model_args = default_cfgs['pretrain_mae_large_patch16_224']

    model = PreTrainMAEVit(
        batch_size=8,
        patch_size=16,
        image_size=224,
        encoder_layers=24,
        encoder_num_heads=16,
        encoder_dim=1024,
        decoder_layers=16,
        decoder_num_heads=32,
        decoder_dim=768,
        mlp_ratio=4,
        masking_ratio=0.75,
        drop_path=0.1,
        initialization=XavierUniform()
        ** kwargs)

    model.dataset_transform = model_args['dataset_transform']

    if pretrained:
        load_pretrained(model, model_args, num_classes=num_classes, in_channels=channels)

    return model
