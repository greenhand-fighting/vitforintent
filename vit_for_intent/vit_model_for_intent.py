"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn 

# droppath函数
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# atention这里只对维度有要求，11*768 和197*768 倒没什么不同的
class Attention(nn.Module):
    def __init__(self,
                 dim,   # embeding dim   这里表示的是 768   
                 num_heads=8,
                 qkv_bias=False,  # 是否使用偏置
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads      # 多头attention的每一个头的维度是 96维
        self.scale = qk_scale or head_dim ** -0.5          # 这是一个因子，是对数值进行放缩的
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)         #  x  ---->   qkv   由x生成qkv          197*768       ---768* 2304   ---->   197*2304    ---分为qkv  --->   197*(768+768+768)   ---8 头attention，每一个头--->  197*(96+96+96)
        self.attn_drop = nn.Dropout(attn_drop_ratio)   # 随机将输入张量中的部分元素设置为0
        self.proj = nn.Linear(dim, dim)                # 前面一个dim表示的是多头已经cancat之后的结果，后面一个dim表示的是 最后影射的维度。
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
      
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
     
        attn = (q @ k.transpose(-2, -1)) * self.scale                                                             # 先对qk进行点乘操作，之后乘以放缩因子。@是矩阵乘法，在多维的情况下，只考虑最后面的两个维度
        attn = attn.softmax(dim=-1)                                                                                         # 在对整体进行softmax，dim=-1就是在每一行上进行softmax 处理
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)                                                      # 这里的意思就是对alue进行加权
        x = self.proj(x)                                                                                                                      # 线性转化
        x = self.proj_drop(x)                                                                                                       
        return x


class Mlp(nn.Module):       # 全链接层，激活函数，dropout ， 全连接层，dropout
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbed1(Mlp):     # 这里就是一个全连接层，从in-features  到 100再到 768
    def __init__(self, in_features, hidden_features=100, out_features=768, act_layer=nn.GELU, drop=0):
        super().__init__(in_features, hidden_features=hidden_features, out_features=out_features, act_layer=act_layer, drop=drop)

# vit就是很多block堆积而成的， block好像也是只对维度有要求，对于是11*768 还是197*768没有区别
class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
       # self.norm1 = norm_layer(dim)    # 进行一次norm                         先不进行layer norm
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        #self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed1, norm_layer=None,
                 act_layer=None,input_feature_dim=4):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(input_feature_dim)    # [1,10,4]  ---> [1,10,768]
        # 使用卷积的方法得到patch emdedding
        num_patches = 10  # 10

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))     # 可训练的参数，直接使用零矩阵进行初始化。形状是(1，1，768)，第一个1 是batch
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))   # position embedding，这里也是可训练的参数，初始化为零矩阵，形状是(1,14*14+1,768)  #-----------------------
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule    drop_path_ratio 这个比例是递增的，在不同的block中使用的比例不同
        self.blocks = nn.Sequential(*[    
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])# 这里是一个列表，里面是一系列的block
        #self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()   # 这里表示不做任何处理

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()  # !！!！!！!！
        self.head_dist = None    
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):    # 刚开始传入的是image。
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]  #-------------------------------------------------------------------------------------------------------------!！!！!！!！!！!！!！!！!
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)      # 相当于对class token进行复制
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]    # class token 和 patchs 进行拼接
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
       # x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])      # 这里就是对feature进行切片，只要class token对应的feature
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):          # !！! 进行forward的时候传入的是image
        x = self.forward_features(x)    
        # self.head_dist==Flase
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)    # 这里是分类头
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)



def create_model(num_classes: int = 3, has_logits: bool = True):
    model = VisionTransformer(
                              embed_dim=768,
                              depth=8,
                              num_heads=8,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model

