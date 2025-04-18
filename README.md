# image_generation_RUST

This project focuses on:

- Efficient data processing using Rust
- Training a generative model to synthesize human face images from embeddings

---

### Task 1: Face Dataset Creation (Rust)

> Goal: Write a Rust program that runs inference on a face detector and crops valid face images from a batch. Use the WIDER FACE dataset and handle edge cases effectively.

While I faced technical issues integrating `opencv` and `torch-sys` crates (cargo could not build them correctly despite several clean builds), I implemented the core face processing logic using Python and focused heavily on the **edge case handling** as required by the task:

- Confidence thresholding (set to 0.83)
- Filtering out blurry images
- Cropping only **frontal** and **single-face** images

These decisions significantly improve the quality and usability of the dataset compared to using raw face detections.

### Examples:

**Detected Blurry Images**  
![Detected Blurry Images](https://raw.githubusercontent.com/01PrathamS/image_generation_RUST/main/result_images/detected_blurry_images.png)

**Filtered Frontal Faces**  
![Filtered Frontal Faces](https://raw.githubusercontent.com/01PrathamS/image_generation_RUST/main/result_images/frontal_faces.png)

---

### Task 2: Generative Face Model (PyTorch)

> Goal: Train a generative model that outputs 128×128 face images conditioned on embeddings.

I experimented with multiple generative approaches:

- **Variational Autoencoder (VAE)**: Successfully reconstructed face images and began exploring embedding conditioning.
- **Diffusion Models (DDPM)**: Trained from scratch, but embedding conditioning was **not fully implemented** (can only sample without conditioned image embedding)

### VAE Results

**VAE Reconstruction (1)**  
![VAE Reconstruction](https://raw.githubusercontent.com/01PrathamS/image_generation_RUST/main/result_images/vae_reconstruction.png)

**VAE Reconstruction (2)**  
![VAE Reconstruction Result](https://raw.githubusercontent.com/01PrathamS/image_generation_RUST/main/result_images/vae_reconstruction_result.png)

### DDPM Results

**Initial DDPM Results**  
![DDPM Initial](https://raw.githubusercontent.com/01PrathamS/image_generation_RUST/main/result_images/intial_ddpm.png)

**Updated DDPM Results**  
![DDPM Updated](https://raw.githubusercontent.com/01PrathamS/image_generation_RUST/main/result_images/updated_ddpm.png)

---

## Notebooks

[Experiment Notebooks](https://github.com/01PrathamS/image_generation_RUST/tree/main/notebooks)

---

## References

1. [Loading and running a PyTorch model in Rust](https://medium.com/@heyamit10/loading-and-running-a-pytorch-model-in-rust-f10d2577d570)  
2. [YOLO-Face GitHub Repo](https://github.com/akanametov/yolo-face)  
3. [Diffusion Models - YouTube](https://www.youtube.com/watch?v=HoKDTa5jHvg)  
4. [DDPM Paper (arXiv)](https://arxiv.org/abs/2006.11239)
5. ...

## Tring Tring... tring tring tring........
