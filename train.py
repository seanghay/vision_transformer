from tqdm import tqdm
from models import VisionTransformer
from datasets import ImageNetDataset
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam

ds_train = ImageNetDataset("./data/train")
ds_val = ImageNetDataset("./data/validation")

model = VisionTransformer(
  image_size=224,
  in_channels=3,
  patch_size=16,
  embedding_dim=224 * 2,
  depth=8,
  n_heads=8,
  qkv_bias=True,
  mlp_ratio=4,
  n_classes=ds_train.n_classes,
)

ce_loss = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001)
train_data_loader = DataLoader(ds_train, batch_size=32)

for epoch in range(5):
  total_loss = 0

  for inputs, targets in tqdm(train_data_loader, desc=f"Epoch: {epoch}"):
    logits = model(inputs)
    loss = ce_loss(logits, targets)
    total_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f"Epoch: {epoch} Loss: {(total_loss / len(train_data_loader)):.4f}")
