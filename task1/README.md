# Face Extraction using YOLOv11 and MTCNN

This project extracts **not blurry**, **frontal** faces from images by combining YOLOv11 for detection and MTCNN for face alignment filtering.

## Edge Cases Handled

- Skips blurry face crops based on FFT high-frequency energy.
- Ignores side-profile or tilted faces using MTCNN keypoints.
- Avoids saving faces with confidence below a defined threshold.

## How to Run

1. Install dependencies

```bash
pip install -r requirements.txt
``` 

2. Download a dataset for testing or provide your  image directory. Ensure it's a single directory containing all the images.
```
kaggle datasets download -d lylmsc/wider-face-for-yolo-training
unzip wider-face-for-yolo-training.zip
```
3. Download the pre-trained YOLOv11 model for face detection:
[https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11m-face.pt](https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11m-face.pt)

4. run Script 
```
python face_crop_blurry.py  --input_dir path/to/images
```

## TADAAAAa...