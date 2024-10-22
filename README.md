
![CLIP-gui-banner2](https://github.com/zer0int/CLIP-XAI-GUI/assets/132047210/208fce6e-221b-4ff3-b7ee-3795a97b4fb6)

### Change 4/May/2024:

- Added AMP (Automatic Mixed Precision); uses torch.cuda.amp / autocast + GradScaler
- ViT models are now much smaller - ViT-L/14 fits into 24 GB VRAM!
- Just do "python run_clipapp-amp.py" to launch the GUI / use AMP for a CLIP 'opinion'.

![before-after](https://github.com/zer0int/CLIP-XAI-GUI/assets/132047210/11b6b703-3f64-42df-9177-31143834d6c2)

-----

## CLIP GUI - XAI app ~ explainable (and guessable) ViT & ResNet

This is a GUI for OpenAI's CLIP ViT and ResNet models, where you can:
- Upload an image, get a CLIP 'opinion' (text) about the image
- --> Gradient Ascent -> optimize text embeddings for cosine similarity with image embedding -> tokenizer -> CLIP 'opinion' words
- Guess where CLIP was 'looking' for a given predicted word by setting a ROI (optional) & see what CLIP was 'looking' at
- --> "GradCAM" - like heatmap of salient features / attention visualization

## Installation & Running

- **Prerequisite**: [OpenAI CLIP](https://github.com/openai/CLIP)
- Check / install `requirements.txt`
- From the console, use "python run_clipapp.py" -> GUI

- Default CLIP ViT-B/32 takes ~15 seconds to generate an 'opinion' (RTX 4090), 4 GB VRAM.
- Gigantic models >> 24 GB VRAM can use NVIDIA Driver CUDA SysMem Fallback Policy to run, but largest models ~ 30 Minutes for 1 opinion (not recommended)
- You can get a smaller model's "opinion" and force that on a bigger model (should work for all >=6 GB VRAM), or add your own words to visualize.
- Check the console to see what CLIP is "MatMulling" about while you wait to get a CLIP opinion.
- Click the image to place a ROI and "guess where CLIP was looking" (gamification, optional).
- Images and texts are saved to the "clipapp" subfolder.
- Check out the examples in "image-examples" to get started with some interesting images (hilarious 'opinion', typographic attack vulnerability, ...).
- Use square images for best results


## Credits / Built On

- [OpenAI / CLIP](https://github.com/openai/CLIP)
- ViT heatmaps built on: [Transformer-MM-Explainability](https://github.com/hila-chefer/Transformer-MM-Explainability)
- ResNet heatmaps built on: [GradCAM Visualization](https://github.com/kevinzakka/clip_playground)
- Original CLIP Gradient Ascent Script: Used with permission by Twitter / X: [@advadnoun](https://twitter.com/advadnoun)
- Special thanks to GPT-4 for coding 90% of the run_clipapp.py =)

## Warning about Bias and Fairness in CLIP Output

CLIP 'opinions' may contain biased rants (especially when non-English text is in the image), slurs, and profanity. Use responsibly / at your own discretion.
For more information, refer to the [CLIP Model Card](https://github.com/openai/CLIP/blob/main/model-card.md).

## Known Issues
- No threading, scripts that invoke models run on main thread (check console to verify thread is not *actually* hanging)

## Examples

![Screenshot 2024-03-23 163731](https://github.com/zer0int/CLIP-XAI-GUI/assets/132047210/17f4bc5f-51e3-4c87-96b5-682a5fcaa794)

![example_git](https://github.com/zer0int/CLIP-XAI-GUI/assets/132047210/170b20e2-9ce1-4b12-bb86-706af89db156)
