# image_generation_RUST

Image generation with Rust

---

## Task 1: Feature Extraction & Preprocessing

To efficiently extract features, I handled several edge cases such as:

- Confidence threshold (0.83)
- Frontal face detection
- Filtering blurry images

**These preprocessing steps significantly improve the quality of generated images — up to 5× better than raw inputs.**

### Examples:

**Detected Blurry Images**  
![Detected Blurry Images](https://raw.githubusercontent.com/01PrathamS/image_generation_RUST/main/result_images/detected_blurry_images.png)

**Filtered Frontal Faces**  
![Filtered Frontal Faces](https://raw.githubusercontent.com/01PrathamS/image_generation_RUST/main/result_images/frontal_faces.png)

---

## Task 2: Generative Modeling

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

 full experimental notebooks here:  
[Experiment Notebooks](https://github.com/01PrathamS/image_generation_RUST/tree/main/notebooks)

---

## References

1. [Loading and running a PyTorch model in Rust](https://medium.com/@heyamit10/loading-and-running-a-pytorch-model-in-rust-f10d2577d570)  
2. [YOLO-Face GitHub Repo](https://github.com/akanametov/yolo-face)  
3. [Diffusion Models - YouTube](https://www.youtube.com/watch?v=HoKDTa5jHvg)  
4. [DDPM Paper (arXiv)](https://arxiv.org/abs/2006.11239)
