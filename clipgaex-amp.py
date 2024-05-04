# -*- coding: utf-8 -*-
"""Ascending CLIPtext.ipynb

2023 GPT-4 & zer0int -- Twitter: @zer0int1
Adaptation of the original notebook by advadnoun, used with explicit permission to publish

# Original Author: Twitter @advadnoun ~ 2021:
Closed Test Ascending CLIPtext.ipynb
This is a notebook for determining descriptions that maximally match an image per CLIP using gradient ascent.

# Top
"""

print("Running CLIP gradient ascent. This can take a minute or two, depending on hardware.\nWhile you wait, here are CLIP's intermediate doings:\n")

#clipmodel = 'ViT-B/32'


import imageio
import torchvision
import PIL.Image
#checkin_step = training_iterations - 1
checkin_step = 25
import os
import sys
import clip
import kornia
import torch
import torch.nn.functional as F
import random
import numpy as np
import argparse
import glob
from multiprocessing import cpu_count
from ldmutil import parallel_data_prefetch
from tqdm import tqdm
from torchvision.transforms import Resize
import warnings
import pickle
import warnings
from colorama import Fore, Style
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="CLIP Gradient Ascent")
parser.add_argument("img_name", type=str, help="Path to the input image")
parser.add_argument("clipmodel", type=str, default='ViT-B/32', help="CLIP model to use")
args = parser.parse_args()

model_to_dims = {
    'RN50': 224, 'RN101': 224, 'ViT-B/32': 224, 'ViT-B/16': 224, 'ViT-L/14': 224,
    'RN50x4': 288, 'RN50x16': 384, 'RN50x64': 448, 'ViT-L/14@336px': 336
}

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Retrieve the input dimension based on the selected CLIP model
clipmodel = args.clipmodel
input_dims = model_to_dims.get(clipmodel, 224)  # Default to 224 if the model is not in the dictionary


training_iterations = 200
batchsize = 12



# Load the clip model architecture
perceptor, preprocess = clip.load(clipmodel, jit=False)
perceptor = perceptor.eval().float()



"""# Def"""

def displ(img, pre_scaled=True):
  img = np.array(img)[:,:,:]
  img = np.transpose(img, (1, 2, 0))
  if not pre_scaled:
    img = scale(img, 48*4, 32*4)
  imageio.imwrite(str(3) + '.png', np.array(img))
  return display.Image(str(3)+'.png')


"""# Internal tweaks"""

def clip_encode_text(gobble, text):
  x = torch.matmul(text, gobble.token_embedding.weight)  # [batch_size, n_ctx, d_model]

  x = x + gobble.positional_embedding
  x = x.permute(1, 0, 2)  # NLD -> LND

  x = gobble.transformer(x)
  x = x.permute(1, 0, 2)  # LND -> NLD
  x = gobble.ln_final(x)

  x = x[torch.arange(x.shape[0]), many_tokens + len(prompt) + 2] @ gobble.text_projection
  #print("Text embeddings shape:", x.shape)

  return x

"""# Settings"""

batch_size = batchsize
many_tokens = 4

# a prompt to use before the learned tokens/words
prompt = clip.tokenize('''''').numpy().tolist()[0]
#print("Tokenized Prompt:", prompt)
prompt = [i for i in prompt if i != 0 and i != 49406 and i != 49407]

sideX = input_dims
sideY = input_dims

# set the image to use
img_path = args.img_name

import os
img_name = os.path.splitext(os.path.basename(img_path))[0]

im = torch.tensor(imageio.imread(img_path).copy()).cuda().unsqueeze(0).permute(0, 3, 1, 2) / 255 # 0,3,1,2 . 255
im = F.interpolate(im, (sideX, sideY))
#print("Image Shape After Preprocessing:", im.shape)

"""
# Setup parameters"""

torch.cuda.empty_cache()

class Pars(torch.nn.Module):
    def __init__(self):
        super(Pars, self).__init__()
        
        st = torch.zeros(batch_size, many_tokens, 49408).normal_()
        self.normu = torch.nn.Parameter(st.cuda())
        self.much_hard = 1000

        self.start = torch.zeros(batch_size, 1, 49408).cuda()
        self.start[:, :, 49406] = 1

        ptt = prompt

        self.prompt = torch.zeros(batch_size, len(ptt), 49408).cuda()
        for jk, pt in enumerate(ptt):
          self.prompt[:, jk, pt] = 1 
        
        self.pad = torch.zeros(batch_size, 77 - (many_tokens + len(prompt) + 1), 49408).cuda()
        self.pad[:, :, 49407] = 1
      
    def forward(self):
      self.soft = F.gumbel_softmax(self.normu, tau=self.much_hard, dim=-1, hard=True)
      fin = torch.cat([self.start, self.prompt, self.soft, self.pad], 1)
      #print("Output shape after forward pass:", fin.shape)
      return fin


lats = Pars().cuda()
mapper = [lats.normu]
optimizer = torch.optim.Adam([{'params': mapper, 'lr': 5}])
eps = 0

nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

augs = torch.nn.Sequential(
    kornia.augmentation.RandomAffine(degrees=10, translate=.1, p=.8).cuda(),
).cuda()

tok = clip.simple_tokenizer.SimpleTokenizer()

bests = {1000:'None', 1001:'None', 1002:'None', 1003:'None', 1004:'None'}

torch.argmax(lats(), 2)[0].clone().detach().cpu().numpy()

"""# Train"""

import warnings
warnings.filterwarnings('ignore')

def augment(into):
  into = augs(into)
  return into

def ascend_txt():
  global im
  iii = nom(augment(im[:,:3,:,:].expand(64, -1, -1, -1)))
  iii = perceptor.encode_image(iii).detach()
  lll = lats()
  tx = clip_encode_text(perceptor, lll)
  return -100*torch.cosine_similarity(tx.unsqueeze(0), iii.unsqueeze(1), -1).view(-1, batch_size).T.mean(1), lll

def train():
    with autocast():
        loss1, lll = ascend_txt()
    loss = loss1.mean()
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss1, lll

def checkin(loss, lll):
    unique_tokens = set()

    these = [tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist()).replace('', '').replace('', '') for kj in range(lll.shape[0])]

    for kj in range(lll.shape[0]):
        if loss[kj] < sorted(list(bests.keys()))[-1]:
            # Remove non-printable characters and replace them with a space
            cleaned_text = ''.join([c if c.isprintable() else ' ' for c in these[kj]])
            bests[loss[kj]] = cleaned_text
            bests.pop(sorted(list(bests.keys()))[-1], None)
            try:
                decoded_tokens = tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist())
                decoded_tokens = decoded_tokens.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
                decoded_tokens = ''.join(c for c in decoded_tokens if c.isprintable())
                print(Fore.WHITE + f"Sample {kj} Tokens: ")
                print(Fore.BLUE + Style.BRIGHT + f"{decoded_tokens}")
            except Exception as e:
                print(f"Error decoding tokens for sample {kj}: {e}")
                continue

    for j, k in zip(list(bests.values())[:5], list(bests.keys())[:5]):
        j = j.replace('<|startoftext|>', '')
        j = j.replace('<|endoftext|>', '')
        j = j.replace('\ufffd', '')
        j = j.replace('.', '')
        j = j.replace(';', '')
        j = j.replace('?', '')
        j = j.replace('!', '')
        j = j.replace('_', '')
        j = j.replace('-', '')
        j = j.replace('\\', '')
        j = j.replace('\'', '')
        j = j.replace('"', '')
        j = j.replace('^', '')
        j = j.replace('&', '')
        j = j.replace('#', '')
        j = j.replace(')', '')
        j = j.replace('(', '')
        j = j.replace('*', '')
        j = j.replace(',', '')

        #print(j, ' ') # not printing them as emojis etc. are non-printable characters in the console
        tokens = j.split()
        unique_tokens.update(tokens)

    with open(f"clipapp/tokens_{img_name}.txt", "w", encoding='utf-8') as f:
        f.write(" ".join(unique_tokens))


def loop():
  for i in range(training_iterations):
    loss, lll = train()
    if i % checkin_step == 0:
      checkin(loss, lll)
loop()