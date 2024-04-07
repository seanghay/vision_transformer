import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
  def __init__(self, image_size, patch_size, in_channels=3, embedding_dim=768):
    super().__init__()

    self.image_size = image_size
    self.patch_size = patch_size

    self.n_patches = int((image_size / patch_size) ** 2)
    self.proj = nn.Conv2d(
      in_channels=in_channels,
      out_channels=embedding_dim,
      kernel_size=self.patch_size,
      stride=patch_size,
    )

  def forward(self, x):
    x = self.proj(x)
    x = x.flatten(2)
    x = x.transpose(1, 2)
    return x


class Attention(nn.Module):
  def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0.0, proj_p=0.0):
    super().__init__()
    self.n_heads = n_heads
    self.dim = dim
    self.head_dim = dim // n_heads
    self.scale = self.head_dim**-0.5
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_p)
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(proj_p)

  def forward(self, x):
    n_samples, n_tokens, dim = x.shape
    qkv = self.qkv(x)
    qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    k_t = k.transpose(-2, -1)
    dp = (q @ k_t) * self.scale

    attn = dp.softmax(dim=-1)
    attn = self.attn_drop(attn)

    weighted_avg = attn @ v
    weighted_avg = weighted_avg.transpose(1, 2)
    weighted_avg = weighted_avg.flatten(2)

    x = self.proj(weighted_avg)
    x = self.proj_drop(x)

    return x


class MultiLayerPerceptron(nn.Module):
  def __init__(self, in_features, hidden_features, out_features, p=0.0):
    super().__init__()
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.act = nn.GELU()
    self.fc2 = nn.Linear(hidden_features, out_features)
    self.drop = nn.Dropout(p)

  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.fc2(x)
    x = self.drop(x)
    return x


class Block(nn.Module):
  def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0.0, attn_p=0.0):
    super().__init__()
    self.norm1 = nn.LayerNorm(dim, eps=1e-6)
    self.attn = Attention(
      dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p
    )
    self.norm2 = nn.LayerNorm(dim, eps=1e-6)
    hidden_features = int(dim * mlp_ratio)
    self.mlp = MultiLayerPerceptron(
      in_features=dim, hidden_features=hidden_features, out_features=dim
    )

  def forward(self, x):
    x = x + self.attn(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x

class VisionTransformer(nn.Module):
  def __init__(
    self,
    image_size=384,
    patch_size=16,
    in_channels=3,
    n_classes=1000,
    embedding_dim=768,
    depth=12,
    n_heads=12,
    mlp_ratio=4.0,
    qkv_bias=True,
    p=0.0,
    attn_p=0.0,
  ):
    super().__init__()
    self.patch_embedding = PatchEmbedding(
      image_size=image_size,
      patch_size=patch_size,
      in_channels=in_channels,
      embedding_dim=embedding_dim,
    )

    self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
    self.pos_embed = nn.Parameter(
      torch.zeros(1, 1 + self.patch_embedding.n_patches, embedding_dim)
    )

    self.pos_drop = nn.Dropout(p=p)
    self.blocks = nn.ModuleList(
      [
        Block(
          dim=embedding_dim,
          n_heads=n_heads,
          mlp_ratio=mlp_ratio,
          qkv_bias=qkv_bias,
          p=p,
          attn_p=attn_p,
        )
        for _ in range(depth)
      ]
    )
    self.norm = nn.LayerNorm(embedding_dim, eps=1e-6)
    self.head = nn.Linear(embedding_dim, n_classes)

  def forward(self, x):
    n_samples = x.shape[0]
    x = self.patch_embedding(x)
    cls_token = self.cls_token.expand(n_samples, -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    x = x + self.pos_embed
    x = self.pos_drop(x)

    for block in self.blocks:
      x = block(x)

    x = self.norm(x)
    cls_token_final = x[:, 0]
    x = self.head(cls_token_final)

    return x


if __name__ == "__main__":
  model = VisionTransformer(
    image_size=224,
    in_channels=3,
    patch_size=16,
    embedding_dim=224 * 2,
    depth=8,
    n_heads=8,
    qkv_bias=True,
    mlp_ratio=4,
    n_classes=100,
  )

  with torch.no_grad():
    inp = torch.rand(1, 3, 224, 224)
    print(inp.dtype)
    print(inp.shape)
    x = model(inp)
    print(x.shape)
    