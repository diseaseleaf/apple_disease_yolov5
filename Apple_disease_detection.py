import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Function to process the image with YOLO
def process_image(file_path):
    img = cv2.imread(file_path)

    # YOLO settings
    whT = 320
    confThreshold = 0.005
    nmsThreshold = 0.05
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

    # Define a dictionary mapping class IDs to descriptions
    classDescriptions = {
        0: "",
        1: "",
        2: "",
        3: "",
    }

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

            # Get the class ID and confidence
            classId = classIds[i]
            confidence = confidence_values[i]

            # Get the description for the class
            if classId in classDescriptions:
                description = classDescriptions[classId]
            else:
                description = "No description available"

            cv2.putText(img, f'{classNames[classId].upper()} {int(confidence * 100)}% - {description}',
                        (rect_x, rect_y - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2)
            print(f'{classNames[classId].upper()} {int(confidence * 100)}% - {description}')

        return img, bbox, classIds

    # Process the image with YOLO
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)
    return findObjects(outputs, img)

# Function to show the reference window
def showReferenceWindow(class_id, classDescriptions, root):
    reference_window = tk.Toplevel(root)
    reference_window.title(f"Class {class_id} Description")

    # Display the description with a Text widget
    description_text = tk.Text(reference_window, wrap="word", height=5, width=40)
    description_text.pack(padx=10, pady=10)

    # Insert the description with a link
    description = classDescriptions.get(class_id, "No description available")
    description_text.insert("1.0", description)

    # Add a link to a website (replace "your_link_here" with the actual link)
    link = "C:/Users/user/Desktop/maskRCNN/Apple model/Home.html"
    description_text.insert("1.0", "\n\nFor prevention, click here:", "link")
    description_text.tag_configure("link", foreground="blue", underline=True)
    description_text.tag_bind("link", "<Button-1>", lambda e: callback(link))

# Callback function for handling mouse click events
def on_click(event, classIds, bbox, classDescriptions, root):
    for box, class_id in zip(bbox, classIds):
        x, y, w, h = box
        if x <= event.x <= (x + w) and y <= event.y <= (y + h):
            showReferenceWindow(class_id, classDescriptions, root)

# Function to handle link clicks
def callback(url):
    import webbrowser
    webbrowser.open_new(url)

# Create a Tkinter root window
root = tk.Tk()
root.withdraw()

# Ask the user to select an image file
file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

# Check if the user selected a file
if file_path:
    img, bbox, classIds = process_image(file_path)

    # Define a dictionary mapping class IDs to descriptions
    classDescriptions = {
        0: "\nThis is Apple scab.",
        1: "\nThis is Apple Black Rot.",
        2: "\nThis is Apple cedar rust",
        3: "\nThis is Healthy.",
        # Add more descriptions for other classes as needed
    }

    # Convert the image to RGB format using PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(img_rgb)

    # Display the main image window using matplotlib
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    plt.title("Click '0' on axis line")

    # Create rectangles for each bounding box
    for box, class_id in zip(bbox, classIds):
        x, y, w, h = box
        rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Connect the mouse click event to the callback function
    fig.canvas.mpl_connect('button_press_event', lambda event, classIds=classIds, bbox=bbox, classDescriptions=classDescriptions, root=root: on_click(event, classIds, bbox, classDescriptions, root))

    plt.show()

    # Show the main Tkinter window
    root.mainloop()

else:
    print("No image selected.")