from textToSpeech import TTS
from keras.models import load_model  # TensorFlow 2.12 and Python 3.8 is required for Keras .h5 to work
from PIL import Image as pImage, ImageOps  # Install pillow instead of PIL
import numpy as np

from tkinter import filedialog


def filepath():
    return filedialog.askopenfilename()


# text to speech engine
tts = TTS()

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image_path = filepath()
image = pImage.open(image_path).convert("RGB")

tts.speak("What is your object's name?")
image_name = input("What is your object's name?: ")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, pImage.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]
confidence_score = str(round(confidence_score*100))+"%"
class_name = class_name[2:]

# Say prediction and confidence score
tts.speak(f"{image_name} is {class_name}. I'm {confidence_score} sure about that!")
