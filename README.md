# image_generation_RUST
image generation with rust


## task1 

to efficiently extracted features i have solved edge cases such as confidence threshold (0.83), frontal face detection, filter blurry images

[e.g. Detected Blurry Images](image_generation_RUST\result_images\detected_blurry_images.png)
[e.g Filtered Frontal Faces](image_generation_RUST\result_images\frontal_faces.png)

## task2 

results with VAE found out here : image_generation_RUST\notebooks\reconstruction_faces_VAE_updated_conv.ipynb

[VAE_reconstruction](image_generation_RUST\result_images\vae_reconstruction.png)
[VAE_reconstruction](image_generation_RUST\result_images\vae_reconstruction_result.png)

results with ddpm : 

[ddpm_initial_results](image_generation_RUST\result_images\intial_ddpm.png)
[ddpm_updated_results](image_generation_RUST\result_images\updated_ddpm.png)


#### References 

1. https://medium.com/@heyamit10/loading-and-running-a-pytorch-model-in-rust-f10d2577d570

2. https://github.com/akanametov/yolo-face

3. [Diffusion Models](https://www.youtube.com/watch?v=HoKDTa5jHvg)

4. [DDPM Paper](https://arxiv.org/abs/2006.11239)
