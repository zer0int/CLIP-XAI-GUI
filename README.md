
![CLIP-gui-banner](https://github.com/zer0int/CLIP-XAI-GUI/assets/132047210/ada8996d-58ea-412d-be94-7b0e64584f1a)

# CLIP GUI - XAI app ~ explainable (and guessable) ViT & ResNet

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
- Some models, such as RN101, not working properly with regard to salient feature heatmaps
- You can edit CLIP's "opinion" and add your own words to the file, but it comes at the expense of some error, alas you need to click twice to select a word to visualize.

## Examples

![example_cmd_git](https://github.com/zer0int/CLIP-XAI-GUI/assets/132047210/42935df4-5298-4193-ad78-0a103e77fb0e)


![example_git](https://github.com/zer0int/CLIP-XAI-GUI/assets/132047210/073b4cbd-057f-48c6-956a-74281c214581)
