# Original notebook from https://github.com/hila-chefer/Transformer-MM-Explainability

import os
import sys
import glob
from PIL import Image
import argparse
import warnings
warnings.filterwarnings('ignore')


# Parse command line arguments
parser = argparse.ArgumentParser(description='Process an image and its corresponding token.')
parser.add_argument('img_name', type=str, help='The name of the image file (with extension)')
parser.add_argument('token_path', type=str, help='The path to the text token file')
parser.add_argument("clipmodel", type=str, help="CLIP model to use")

parser.add_argument('--roi_x', type=int, default=0, help='X coordinate of the ROI')
parser.add_argument('--roi_y', type=int, default=0, help='Y coordinate of the ROI')
parser.add_argument('--roi_width', type=int, default=100, help='Width of the ROI')
parser.add_argument('--roi_height', type=int, default=100, help='Height of the ROI')

args = parser.parse_args()


# Use the arguments
image_name = args.img_name
token_path = args.token_path
clipmodel = args.clipmodel

# Assuming the rest of your script here...

heatmap_folder = 'clipapp'


def show_image_relevance(image_relevance, image, orig_image, img_path):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return image_relevance
    
import torch
import CLIP.clip as clip
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from captum.attr import visualization

#@title Control context expansion (number of attention layers to consider)
#@title Number of layers for image Transformer
start_layer =  -1#@param {type:"number"}

#@title Number of layers for text Transformer
start_layer_text =  -1#@param {type:"number"}

def interpret(image, texts, model, device, start_layer=start_layer, start_layer_text=start_layer_text):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, logits_per_text = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1:
      # calculate index of last layer
      start_layer = len(image_attn_blocks) - 1

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]


    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

    if start_layer_text == -1:
      # calculate index of last layer
      start_layer_text = len(text_attn_blocks) - 1

    num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_text:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R_text = R_text + torch.bmm(cam, R_text)
    text_relevance = R_text

    return text_relevance, image_relevance

def show_image_relevance(image_relevance, image, orig_image):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(orig_image);
    axs[0].axis('off');

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    axs[1].imshow(vis);
    axs[1].axis('off');
    return image_relevance

from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

def show_heatmap_on_text(text, text_encoding, R_text):
  CLS_idx = text_encoding.argmax(dim=-1)
  R_text = R_text[CLS_idx, 1:CLS_idx]
  text_scores = R_text / R_text.sum()
  text_scores = text_scores.flatten()
  #print(text_scores)
  text_tokens=_tokenizer.encode(text)
  text_tokens_decoded=[_tokenizer.decode([a]) for a in text_tokens]
  vis_data_records = [visualization.VisualizationDataRecord(text_scores,0,0,0,0,0,text_tokens_decoded,1)]
  #visualization.visualize_text(vis_data_records)


clip.clip._MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt", 
}

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(clipmodel, device=device, jit=False)

def show_image_relevance(image_relevance, image, orig_image, img_path):
    # the function body here...
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam
    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    image_relevance_masking = image_relevance
    return vis
    
def get_image_relevance(image_relevance, image):
    # Just process 'image_relevance' for the purpose of creating the binary mask
    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    return image_relevance


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


# Directly use the image name and token path provided
img_file = args.img_name  # Directly use the image file name from command line args
token_file = args.token_path  # Directly use the token file path from command line args

# Process the specified image and token file
try:
    # Open and process the specified image
    img = Image.open(img_file)
    img = img.convert('RGB')  # Convert to RGBA format
    
    # Directly process the image without saving or converting, as it's already specified by the user
    img_processed = preprocess(img).unsqueeze(0).to(device)
    
    # Read the corresponding token
    with open(token_file, 'r') as f:
        tokens = f.read().split()

    print(f"Processing {img_file} tokens...")

    # Process each token for the given image
    for token in tokens:
        texts = [token]
        text = clip.tokenize(texts).to(device)

        # Run the model
        R_text, R_image = interpret(model=model, image=img_processed, texts=text, device=device)
        batch_size = text.shape[0]
        for i in range(batch_size):
            show_heatmap_on_text(texts[i], text[i], R_text[i])
            vis = show_image_relevance(R_image[i], img_processed, orig_image=img, img_path=img_file)

            # Save the heatmap image with the token in the filename
            heatmap_filename = f"{heatmap_folder}/{os.path.splitext(os.path.basename(img_file))[0]}_{token}.png"
            vis = cv2.resize(vis, (224, 224), interpolation=cv2.INTER_AREA)
            vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # Convert the image to BGR format for OpenCV
            cv2.imwrite(heatmap_filename, vis)
except IOError as e:
    print(f"Error processing file {img_file}: {e}")
    
    
# Assuming you've already computed `image_relevance` as shown in your provided code
# Obtain 'image_relevance' specifically for binary mask creation
image_relevance_for_mask = get_image_relevance(R_image[i], img_processed)

# Proceed with creating the binary mask and saving it as before
threshold = 0.4  # Define or adjust your threshold value here
binary_mask = image_relevance_for_mask >= threshold

# Convert the binary mask to an image format (PIL Image) for saving
mask_image = Image.fromarray(np.uint8(binary_mask * 255), 'L')

mask_image_resized = mask_image.resize((224, 224), Image.LANCZOS)

# Save the binary mask image to the specified directory
binary_mask_filename = f"{heatmap_folder}/tmp/binary_mask_{os.path.splitext(os.path.basename(args.img_name))[0]}.png"
#mask_image.save(binary_mask_filename)
mask_image_resized.save(binary_mask_filename)


print(f"...Done.")
