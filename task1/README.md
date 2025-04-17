# Face Extraction using YOLOv11 and MTCNN

This project extracts **not blurry**, **frontal** faces from images using a combination of YOLOv11 for detection and MTCNN for face alignment filtering.

## Features

-   Uses **YOLOv11** for face detection
-   Uses **MTCNN** to check if the detected face is **frontal**
-   Uses **FFT (Fast Fourier Transform)** to discard **blurry images**
-   Automatically **scales** bounding boxes for better face crops
-   Filters detections using a **confidence threshold**
-   Outputs cropped face images to a directory

## Edge Cases Handled

-   Skips blurry face crops based on FFT high-frequency energy
-   Ignores side-profile or tilted faces using MTCNN keypoints
-   Avoids saving faces with confidence below a defined threshold
-   Prevents out-of-bound coordinates for face cropping

## 🚀 How to Run

1. Install dependencies

```bash
pip install -r requirements.txt
``` 

2. Download Dataset (Optional - Provide your own image directory)You can download a dataset for testing. Make sure it 's single directory containing all the images 
```
kaggle datasets download -d lylmsc/wider-face-for-yolo-training
unzip wider-face-for-yolo-training.zip
```
3. Download YOLOv11 Face Detection ModelDownload the pre-trained YOLOv11 model for face detection:
[https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11m-face.pt](https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11m-face.pt)

4. run Script 
```
python face_crop_blurry.py  --input_dir path/to/images
```