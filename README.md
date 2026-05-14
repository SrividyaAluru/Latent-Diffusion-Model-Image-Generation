# Hot-Dog Generative Model — VAE + Latent Diffusion (DDPM)

A from-scratch implementation of a **Variational Autoencoder (VAE)** and a **Latent Diffusion Model (DDPM)** that generates hot-dog images from pure Gaussian noise. The DDPM achieved a **hot-dog detector confidence of 0.93**.

---

## Results

### Best Generated Image (Confidence: 0.93)
![Best generated hot-dog](best_hotdog.png)


### Sample Grid from DDPM
![Generated image grid](generated_grid.png)

---

## Part 1 — Variational Autoencoder

### What a VAE does

A VAE learns two things simultaneously: how to compress an image into a compact latent representation, and how to generate plausible new images by sampling from that latent space. The key idea is that instead of encoding an image to a single point in latent space, the encoder outputs the **mean (μ) and log-variance (log σ²)** of a Gaussian distribution. A sample is drawn from that distribution using the **reparameterization trick** — `z = μ + ε·σ` where `ε ~ N(0, I)` — which keeps the operation differentiable so gradients can flow through it during training. The decoder then reconstructs the image from `z`.

The loss has two terms:
- **Reconstruction loss (NLL):** How faithfully the decoder reproduces the input
- **KL divergence:** Pushes the latent distribution toward `N(0, I)`, ensuring the latent space is smooth and structured enough to sample from

### Architecture

Built a fully convolutional VAE (under 1M parameters):

**Encoder**
```
Input (1×28×28 / 3×28×28)
→ Conv2d(→32, k=3, s=2) + BatchNorm + LeakyReLU    # 14×14
→ Conv2d(→64, k=3, s=2) + BatchNorm + LeakyReLU    # 7×7
→ Flatten → Linear → μ, log σ²   (latent_dim = 20)
```

**Decoder**
```
z (latent_dim=20)
→ Linear → Reshape (64×7×7)
→ ConvTranspose2d(→64, s=2) + BatchNorm + LeakyReLU   # 14×14
→ ConvTranspose2d(→32, s=2) + BatchNorm + LeakyReLU   # 28×28
→ Conv2d(→channels)
```

### Training

| Setting | Value |
|---------|-------|
| Latent dim | 20 |
| MNIST epochs | 16 |
| Hot-dog epochs | 30 |
| Learning rate | 1e-4 (hot-dog) |
| Batch size | 128 / 64 |
| β values tested | 1, 2, 5 |
| Optimizer | Adam |

Trained on MNIST first to validate the architecture, then transferred to 28×28 hot-dog images. Experimented with β ∈ {1, 2, 5} to study the KL weighting trade-off — higher β produces a smoother, more structured latent space at the cost of reconstruction sharpness.

### Why VAE sampling alone isn't enough

Even with a well-trained VAE, random samples from the latent space are poor. KL regularisation pushes the posterior toward a Gaussian, but complex data distributions are hard to compress into a perfectly smooth Gaussian — leaving gaps in latent space where decoded samples are incoherent. The decoder also performs reconstruction in a single forward pass with no ability to self-correct. This is what motivated the move to latent diffusion.

---

## Part 2 — Latent Diffusion Model (DDPM)

### What a DDPM does

A DDPM learns to reverse a **forward diffusion process** that progressively destroys an image by adding Gaussian noise over T timesteps. At timestep t:

```
q(xₜ | xₜ₋₁) = N(xₜ; √(1−βₜ)·xₜ₋₁, βₜ·I)
```

In closed form, you can jump straight to any noise level:

```
q(xₜ | x₀) = N(xₜ; √ᾱₜ·x₀, (1−ᾱₜ)·I)
```

where ᾱₜ = ∏(1−βₛ) for s=1..t. A U-Net is trained to predict the noise `ε` added at each step, and generation works by starting from pure noise and iteratively denoising over all T steps.

Working in **latent space** rather than pixel space is the key practical choice — the SD2 VAE compresses 112×112 images down to 4×14×14 latents, making training feasible on a single GPU.

### Noise Schedule

Linear schedule from β₁ = 1e-5 to β₂ = 0.005 over T = 1000 steps:

```
βₜ = β₁ + (β₂ − β₁)·t/T
```

Pre-computed tensors registered as model buffers: `alpha_t`, `alphabar_t`, `sqrtab`, `sqrtmab`, `oneover_sqrta`, `mab_over_sqrtmab`, `sqrt_beta_t`.

### Architecture — SimpleUNet

A U-Net with **sinusoidal timestep embeddings** to condition each layer on the current noise level:

```
Time embedding: SinusoidalPositionEmbeddings(256) → Linear(256,256) → ReLU

Downsampling:
  conv0:   4  → 112 channels
  down1:  112 → 224 channels
  down2:  224 → 448 channels

Bottleneck: 448 channels

Upsampling (with skip connections from down path):
  up1:  448 → 224 channels
  up2:  224 → 112 channels
  out:  112 →   4 channels  (matches latent channels)
```

### Training Configuration

| Setting | Value |
|---------|-------|
| Timesteps (T) | 1000 |
| β range | (1e-5, 0.005) |
| Optimizer | AdamW |
| Image size | 112×112 |
| Latent shape | 4×14×14 |
| Time embedding dim | 256 |
| Channel widths | (112, 224, 448) |

### SD2 VAE Backbone

Used the frozen VAE from `Manojb/stable-diffusion-2-base` (HuggingFace) as the image encoder/decoder. The DDPM is trained entirely in latent space — images are encoded once per batch, the U-Net learns to denoise the latent representations, and the VAE decoder reconstructs the final pixel-space image at inference time.

### Hot-Dog Detector Result

The best generated image scored **0.93 confidence** on the provided hot-dog detector (trained on Roboflow Hot-Dog Detection + Food-101 datasets), placing it in the top-5 predictions.

---

## Stack

- PyTorch, torchvision
- HuggingFace `diffusers` (SD2 VAE backbone)
- Trained on GPU (DoC GPUDOJO cluster)

---

## References

- [Ho et al. (2020) — DDPM](https://arxiv.org/abs/2006.11239)
- [Rombach et al. (2022) — Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Lilian Weng — Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Kingma & Welling (2013) — VAE](https://arxiv.org/abs/1312.6114)# Latent-Diffusion-Model-Image-Generation
New Hot Dog image generation from scratch and implementing DDPM paper
