# Sal-Guide-Diffusion
![1703167429749](https://github.com/superlnX/Sal-Guide-Diffusion/assets/154779934/284331c6-8ca0-4fa5-aaef-c5cff02f6ff3)

The Pytorch implementation of “Sal-Guide Diffusion: Saliency Maps Guide Emotional Image Generation through Adapter”.

This is a preprint version for 2024 ICME.

## Requirements 

- transformers==4.19.2
- diffusers==0.11.1
- basicsr==1.4.2
- einops==0.6.0
- omegaconf==2.3.0
- pytorch_lightning==1.5.9
- opencv-python
- open-clip-torch
- safetensors

## Usage

- Place the dataset and corresponding saliency map into the dataset folder.
- Modify the path of the images in train.py, and then run it.
- Running test.py can obtain a batch of images generated by stable diffusion and our method under the same seed.


Our pretrained model is coming soon.
