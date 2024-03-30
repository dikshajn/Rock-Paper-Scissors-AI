import cv2
import numpy as np
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
from keras.models import load_model
import os
import sys

# Function to gather data images with a particular label
def gather_images(label_name, num_samples):
    IMG_SAVE_PATH = 'image_data'
    IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, label_name)

    try:
        os.mkdir(IMG_SAVE_PATH)
    except FileExistsError:
        pass
    try:
        os.mkdir(IMG_CLASS_PATH)
    except FileExistsError:
        print("{} directory already exists.".format(IMG_CLASS_PATH))
        print("All images gathered will be saved along with existing items in this folder")

    cap = cv2.VideoCapture(0)
    start = False
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if count == num_samples:
            break

        cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)

        if start:
            roi = frame[100:500, 100:500]
            save_path = os.path.join(IMG_CLASS_PATH, '{}.jpg'.format(count + 1))
            cv2.imwrite(save_path, roi)
            count += 1

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Collecting {}".format(count),
                    (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Collecting images", frame)

        k = cv2.waitKey(10)
        if k == ord('a'):
            start = not start

        if k == ord('q'):
            break

    print("\n{} image(s) saved to {}".format(count, IMG_CLASS_PATH))
    cap.release()
    cv2.destroyAllWindows()

# Define the label name and number of samples
label_name = "rock"  # Change label name as needed
num_samples = 100  # Change the number of samples as needed

# Gather images
gather_images(label_name, num_samples)

# Load images from the directory
dataset = []
for directory in os.listdir('image_data'):
    path = os.path.join('image_data', directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227, 227))
        dataset.append([img, directory])

# Convert labels to one-hot encoded format
data, labels = zip(*dataset)
labels = np.array([0 if label == label_name else 1 for label in labels])
labels = np_utils.to_categorical(labels)

# Define the model architecture
model = Sequential([
    SqueezeNet(input_shape=(227, 227, 3), include_top=False),
    Dropout(0.5),
    Convolution2D(2, (1, 1), padding='valid'),
    Activation('relu'),
    GlobalAveragePooling2D(),
    Activation('softmax')
])

# Compile the model
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(np.array(data), labels, epochs=10)

# Save the model
model.save("rock-paper-scissors-model.h5")

# Use the trained model to predict the gesture in a given image file
def predict_gesture(filepath):
    REV_CLASS_MAP = {
        0: "rock",
        1: "paper",
        2: "scissors",
        3: "none"
    }

    def mapper(val):
        return REV_CLASS_MAP[val]

    model = load_model("rock-paper-scissors-model.h5")

    # prepare the image
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))

    # predict the move made
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    move_name = mapper(move_code)

    print("Predicted: {}".format(move_name))

# Use the function to predict gesture in a given image file
if len(sys.argv) > 
