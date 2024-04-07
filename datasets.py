import torch
import os
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2


class ImageNetDataset(Dataset):
  def __init__(self, image_dir, image_size=224):
    super().__init__()
    self.transform = v2.Compose(
      [
        v2.Resize((image_size, image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
      ]
    )

    self.image_size = image_size
    self.items = []
    classes = set()

    for image_file in tqdm(glob(os.path.join(image_dir, "**/*.jpg"))):
      label_name = os.path.basename(os.path.dirname(image_file))
      label_id = int(label_name) - 1
      classes.add(label_id)
      img = read_image(image_file)
      img = self.transform(img)
      self.items.append((img, label_id))

    self.n_classes = len(classes)

  def __len__(self):
    return len(self.items)

  def __getitem__(self, index):
    image, label = self.items[index]
    return image, label


if __name__ == "__main__":
  ds = ImageNetDataset("./data/train")
  print(f"{len(ds)=}")
  print(ds[0][0].shape)
  print(ds[0][0].dtype)
