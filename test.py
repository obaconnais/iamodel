from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
from class_plot import plot_bboxes

model = YOLO("yolov8n.pt")


results = model.predict("./photos/P&O.jpg")
