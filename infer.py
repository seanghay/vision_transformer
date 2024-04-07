from models import VisionTransformer
import torch

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
  x = model(inp)
  print(x.shape)