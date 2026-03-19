import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as xet
from glob import glob
from shutil import copy
import cv2
import pytesseract as pt
from skimage import io

# 1. Data Preparation for YOLO
def parse_xml_for_yolo(xml_folder, images_folder):
    path = glob(os.path.join(xml_folder, '*.xml'))
    records = []
    for filename in path:
        parser = xet.parse(filename).getroot()
        name = parser.find('filename').text
        img_path = os.path.join(images_folder, name)
        size = parser.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        obj = parser.find('object')
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        center_x = (xmax + xmin) / (2 * width)
        center_y = (ymax + ymin) / (2 * height)
        bb_width = (xmax - xmin) / width
        bb_height = (ymax - ymin) / height
        records.append([img_path, width, height, center_x, center_y, bb_width, bb_height])
    df = pd.DataFrame(records, columns=['filename', 'width', 'height', 'center_x', 'center_y', 'bb_width', 'bb_height'])
    return df

def prepare_yolo_folders(df, yolo_train_folder, yolo_test_folder, split_idx=200):
    os.makedirs(yolo_train_folder, exist_ok=True)
    os.makedirs(yolo_test_folder, exist_ok=True)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    for subset, folder in zip([df_train, df_test], [yolo_train_folder, yolo_test_folder]):
        for _, row in subset.iterrows():
            fname = row['filename']
            image_name = os.path.basename(fname)
            txt_name = os.path.splitext(image_name)[0]
            dst_image_path = os.path.join(folder, image_name)
            dst_label_file = os.path.join(folder, txt_name + '.txt')
            copy(fname, dst_image_path)
            label_txt = f"0 {row['center_x']} {row['center_y']} {row['bb_width']} {row['bb_height']}"
            with open(dst_label_file, 'w') as f:
                f.write(label_txt)

# 2. YOLOv5 Training and Export
# Run these commands in terminal or subprocess:
# !git clone https://github.com/ultralytics/yolov5
# !pip install -r ./yolov5/requirements.txt
# !python ./yolov5/train.py --data ./data.yaml --cfg ./yolov5/models/yolov5s.yaml --batch-size 8 --name Model --epochs 100
# !python ./yolov5/export.py --weights ./yolov5/runs/train/Model/weights/best.pt --include torchscript onnx

# 3. Prediction & OCR using exported YOLOv5 model
def yolo_predict_and_ocr(image_path, onnx_model_path):
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 640
    net = cv2.dnn.readNetFromONNX(onnx_model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    img = io.imread(image_path)
    row, col, d = img.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = img
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    boxes, confidences = [], []
    x_factor = input_image.shape[1] / INPUT_WIDTH
    y_factor = input_image.shape[0] / INPUT_HEIGHT
    for row in detections:
        confidence = row[4]
        if confidence > 0.4:
            class_score = row[5]
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
    for i in indices:
        x, y, w, h = boxes[i]
        roi = img[y:y+h, x:x+w]
        text = pt.image_to_string(roi)
        print(f"Predicted Plate Text: {text.strip()}")
        return roi, text.strip()
    print("No plate detected.")
    return None, None

if __name__ == "__main__":
    # 1. Prepare YOLO data
    df = parse_xml_for_yolo('./xml_folder', './images_folder')  # <-- set your XML and images folder
    prepare_yolo_folders(df, './yolov5/data_images/train', './yolov5/data_images/test')

    # 2. Train YOLOv5 and export model (run commands in terminal as comments above)

    # 3. Predict and OCR
    roi, text = yolo_predict_and_ocr('./test_image.jpg', './yolov5/runs/train/Model/weights/best.onnx')  # <-- set your test image and model path