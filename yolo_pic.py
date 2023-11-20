import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Create a Tkinter root window (it won't be shown)
root = tk.Tk()
root.withdraw()

# Ask the user to select an image file
file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

# Check if the user selected a file
if file_path:
    # Read the selected image
    img = cv2.imread(file_path)

    # YOLO settings
    whT = 320
    confThreshold = 0.5
    nmsThreshold = 0.3
    classesFile = 'C:/Users/user/Desktop/maskRCNN/Apple model/apple.names'
    modelConfiguration = 'C:/Users/user/Desktop/maskRCNN/Apple model/apple-tiny.cfg'
    modelWeights = 'C:/Users/user/Desktop/maskRCNN/Apple model/apple-tiny.weights'

    # Read the class names
    classNames = []
    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    # Load YOLO model
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def findObjects(outputs, img):
        hT, wT, cT = img.shape
        bbox = []
        classIds = []
        confidence_values = []

        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    classIds.append(classId)
                    confidence_values.append(float(confidence))
        print(len(bbox))
        indices = cv2.dnn.NMSBoxes(bbox, confidence_values, confThreshold, nmsThreshold)

        for i in indices:
            box = bbox[i]
            rect_x, rect_y, rect_width, rect_height = 20, 50, 200, 200
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 0, 0), 2)
            cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confidence_values[i] * 100)}%',
                        (rect_x, rect_y - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2)
            print(f'{classNames[classIds[i]].upper()} {int(confidence_values[i] * 100)}%')

    # Process the image with YOLO
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    cv2.imshow('Output', img)
    #cv2.resizeWindow('Output',1920,1080)
    cv2.waitKey(0)
else:
    print("No image selected.")
