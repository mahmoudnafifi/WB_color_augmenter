################################################################################
# Copyright (c) 2019-present, Mahmoud Afifi
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
#
# Please, cite the following paper if you use this code:
# Mahmoud Afifi and Michael S. Brown. What else can fool deep learning?
# Addressing color constancy errors on deep neural network performance. ICCV,
# 2019
#
# Email: mafifi@eecs.yorku.ca | m.3afifi@gmail.com
################################################################################

import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from WBAugmenter import WBEmulator as wbAug
from PIL import Image
from os.path import join
from os import listdir
from random import random
from torch.utils.data import DataLoader
import urllib.request


################################################################################
### BASIC DATASET with WB augmentation
################################################################################

class BasicDataset(Dataset):
  def __init__(self, img_dir, aug_prob=0.5):
    self.img_dir = img_dir
    ext = ['.png', '.jpg']
    self.imgfiles = [join(img_dir, file) for file in listdir(img_dir) if
                     file.endswith(ext[0]) or file.endswith(ext[1])]
    self.wb_color_aug = wbAug.WBEmulator()
    if aug_prob > 0.0:
      self.target_dir = self.wb_color_aug.precompute_mfs(self.imgfiles)
    else:
      self.target_dir = None
    self.aug_prob = aug_prob
    # VGG preprocessing .. change it accordingly
    self.preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])

  def __len__(self):
    return len(self.imgfiles)

  def delete_mapping_funcs(self):
    if self.target_dir is None:
      return
    self.wb_color_aug.delete_precomputed_mfs(self.imgfiles,
                                             target_dir=self.target_dir)

  def __getitem__(self, i):
    img_file = self.imgfiles[i]
    if random() < self.aug_prob:
      img = self.wb_color_aug.open_with_wb_aug(img_file, self.target_dir)
    else:
      img = Image.open(img_file)
    img = self.preprocess(img)
    return {'image': img}


################################################################################

dataset_dir = '../images'
augmentation_probability = 1.0
model = models.vgg19(pretrained=True).to('cuda')
model.eval()

# Download ImageNet labels
url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
urllib.request.urlretrieve(url, 'imagenet_classes.txt')

dataset = BasicDataset(dataset_dir, aug_prob=augmentation_probability)
data_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

# Read imagenet categories
with open("imagenet_classes.txt", "r") as f:
  categories = [s.strip() for s in f.readlines()]

for epoch in range(100):
  for batch in data_loader:
    imgs = batch['image'].to('cuda')
    output = model(imgs)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # Show top categories per image
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    print(categories[top1_catid], top1_prob.item())

dataset.delete_mapping_funcs()