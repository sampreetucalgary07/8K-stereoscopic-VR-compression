# 8K Stereoscopic VR Compression

![Architecture](img/Architecture.png)

This repository accompanies our research project titled **"Compression and Transmission of 8K Stereoscopic VR using VAE-GAN Latents and Standard Encoders."** It includes the core modules and configurations required to replicate our experiments. If you require a pretrained model, please email us at: `sampreet.vaidya@ucalgary.ca`.

## Overview

This project investigates the compression and transmission of 8K stereoscopic VR content using VAE-GAN latents integrated with standard encoders. Our approach is designed to achieve high-quality compression and efficient streaming, making it suitable for immersive applications.

<!-- ## Key Features

- **8K Stereoscopic VR Compression:** Combining Variational Autoencoders (VAE) with Generative Adversarial Networks (GAN) for latent-based video compression.
- **Support for Standard Encoders:** Latent space compressed representations are encoded with conventional codecs like H.264 and H.265.
- **Client-Side Reconstruction:** Demonstrates the perceptual quality of reconstructed VR frames compared to conventional methods.

--- -->

## Hardware Requirements

Experiments were conducted on cloud servers with **NVIDIA A100 GPUs**. Due to the computational complexity of 8K frame processing, similar hardware is strongly recommended for optimal performance.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/sampreetucalgary07/8K-stereoscopic-VR-compression.git
cd 8K-stereoscopic-VR-compression
```

2. Install dependencies:

```bash
pip3 install -r requirements.txt

```

3. Download the 8K stereo dataset from this [paper](https://www.researchgate.net/publication/379125834_Towards_evaluation_of_immersion_visual_comfort_and_exploration_behaviour_for_non-stereoscopic_and_stereoscopic_360_videos). Ensure you have sufficient storage space available (the dataset exceeds 15 GB).

4. Place the videos in `data/Videos` while all the corresponding preprocessing steps such as extraction of frames (top-down) should be in `data/video_dataset`. The empty folders have been already created.

5. Edit the configuration files under the `configs` folder (e.g., `configs/E2_8K_basketball.yaml`) based on your experimental requirements. Default settings are available for corresponding video scenarios.

6. Run `train_vae_model.py `. To automate multiple experiments, use the dynamic script: such as ``` dynamic_scripts_video_process.py to carry video preprocessing on all the files.

## Results

Below is the visual comparison of client-side reconstructions. The H.265 encoded frame (second-left) is perceptually very similar to our losslessly transmitted latent-based reconstruction (second-right). Size of the latent frame (rightmost) is equal to red block in the original frame (leftmost).

![visual-result](img/visual-result.png)

### Quantitative Metrics

| **8K Scene**  | **H.264** | **H.265** | **Our Method** | **H.264** | **H.265** | **Our Method** |
|----------------|-----------|-----------|----------------|-----------|-----------|----------------|
|                |           |           |                | **Rate-Distortion Score** | **Rate-Distortion Score** | **Rate-Distortion Score** |
| **Basketball** | 81.83     | 60.89     | **46.34**      | 0.505     | 0.209     | **0.210**      |
| **Park**       | 256.3     | 239.99    | **84.75**      | 0.510     | 0.457     | **0.105**      |
| **Football**   | 108.36    | 90.74     | **44.49**      | 0.509     | 0.372     | **0.054**      |
| **Grass**      | 408.45    | 373.63    | **85.92**      | 0.515     | 0.451     | **0.055**      |
| **Sunny**      | 67.84     | 53.15     | **48.25**      | 0.505     | 0.130     | **0.120**      |


## Citation

If you find this work useful in your research, please cite our paper.

@inproceedings{
title={Compression and Transmission of 8K Stereoscopic
VR using VAE-GAN Latents and Standard Encoders},
author={Vaidya, Sampreet and Abou-Zeid, Hatem and Krishnamurthy, Diwakar},
year={2025},
conference={2025 IEEE Wireless Communications and Networking Conference}, }

## References

1. L. V. Academy, "Monoscopic vs. stereoscopic 360° VR", link, accessed 2023-08-21.
2. Zhao et al., "Virtual reality gaming on the cloud: A reality check", IEEE GLOBECOM 2021.
3. Liu et al., "Deep learning in latent space for video prediction and compression", CVPR 2021.
4. Chen et al., "LSVC: A learning-based stereo video compression framework", CVPR 2022.
5. Hu et al., "One-click upgrade from 2D to 3D: Sandwiched RGB-D video compression for stereoscopic teleconferencing", CVPR 2024 Workshop.
6. Birman et al., "Overview of research in the field of video compression using deep neural networks", Multimedia Tools and Applications, 2020.
7. Meta, "Encoding immersive videos for Meta Quest 2", link, accessed 2023-11-16.
8. Fremerey et al., "Towards evaluation of immersion, visual comfort and exploration behaviour for non-stereoscopic and stereoscopic 360° videos", IEEE ISM 2023.
9. Vaidya et al., "Transfer learning for online prediction of virtual reality cloud gaming traffic", IEEE GLOBECOM 2023.
10. Hou et al., "Low-latency neural stereo streaming", CVPR 2024.
11. Kingma and Welling, "Auto-encoding variational bayes", 2013.
12. Zhang et al., "The unreasonable effectiveness of deep features as a perceptual metric", CVPR 2018.
13. Isola et al., "Image-to-image translation with conditional adversarial networks", CVPR 2017.
14. Theis et al., "Lossy compression with gaussian diffusion", 2022.
15. Hewage, "3D video processing and transmission fundamentals", Bookboon, 2014.
