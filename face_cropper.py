import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import torch


# === CONFIGURATION ===
CONFIDENCE_THRESHOLD = 0.83  ## probability threshold for face detection
SCALE = 1.7  # padding around detected face
MAX_IMAGES = 1000  ## number of images to process
FFT_BLUR_THRESHOLD = 1e7  ## threashold for blurry images 




def load_image_paths(input_dir: str, max_images: int = MAX_IMAGES) -> list:
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.png'))
    ])
    return [os.path.join(input_dir, f) for f in image_files[:max_images]]


def load_model(model_path: str = "yolov11m-face.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path).to(device)
    return model, device


def is_sharp_fft(image_np, threshold=FFT_BLUR_THRESHOLD):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = gray.shape
    center = (h // 2, w // 2)
    mask = np.ones((h, w), np.uint8)
    mask[center[0]-30:center[0]+30, center[1]-30:center[1]+30] = 0
    high_freq_energy = np.sum(magnitude * mask)

    return high_freq_energy >= threshold


def extract_faces_from_image(
    image_path: str,
    model,
    device,
    output_folder: str,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    scale: float = SCALE,
    fft_threshold: float = FFT_BLUR_THRESHOLD
) -> int:
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    results = model.predict(
        source=image_np,
        verbose=False,
        device=device.index if device.type == "cuda" else None
    )
    result = results[0]

    if result.boxes is None:
        return 0

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    saved_count = 0

    for face_id, (box, conf) in enumerate(zip(boxes, confidences)):
        if conf < confidence_threshold:
            continue

        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        new_w = box_w * scale
        new_h = box_h * scale
        x1_new = max(0, int(center_x - new_w / 2))
        y1_new = max(0, int(center_y - new_h / 2))
        x2_new = min(w, int(center_x + new_w / 2))
        y2_new = min(h, int(center_y + new_h / 2))

        face_crop = image_np[y1_new:y2_new, x1_new:x2_new]

        if not is_sharp_fft(face_crop, threshold=fft_threshold):
            continue

        filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_folder, f"{filename}_face_{face_id}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
        saved_count += 1

    return saved_count


def process_images(input_dir: str):
    output_dir = "cropped_faces"
    os.makedirs(output_dir, exist_ok=True)

    image_paths = load_image_paths(input_dir)
    model, device = load_model()

    total_faces = 0
    for img_path in tqdm(image_paths, desc="Extracting faces"):
        count = extract_faces_from_image(
            image_path=img_path,
            model=model,
            device=device,
            output_folder=output_dir
        )
        total_faces += count

    print(f"\n✅ Done! Total sharp faces saved: {total_faces}")
    print(f"📁 Output folder: {output_dir}")
    return output_dir



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop sharp faces from images using YOLO and FFT blur detection.")
    parser.add_argument("input_dir", type=str, help="Path to input image directory")
    args = parser.parse_args()

    process_images(args.input_dir)


    ## python face_cropper.py /path/to/images_dir
