import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import os

#Load the class indices
class_indices_path = 'C:\\Users\\danie\\Documents\\GitHub\\850_2\\class_indices.json'
with open(class_indices_path, 'r') as class_file:
    class_indices = json.load(class_file)

#Load the trained model
model_path = 'C:\\Users\\danie\\Documents\\GitHub\\850_2\\my_model1.keras'  # Update with the actual path to your saved model
model = load_model(model_path)

#Function to preprocess a single image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(100, 100))
    img_tensor = image.img_to_array(img)  #convert the image to a numpy array
    img_tensor = np.expand_dims(img_tensor, axis=0)  #add batch dimension
    img_tensor /= 255.  #rescale by 1/255
    return img_tensor

#Function to predict the class of a single image and return probabilities 
def predict_image_class(model, image_path):
    img_tensor = preprocess_image(image_path)
    probabilities = model.predict(img_tensor)
    return probabilities

#test image paths
test_image_paths = {
    'Medium': 'C:\\Users\\danie\\Documents\\GitHub\\850\\data\\Data\\Test\\Medium\\Crack__20180419_06_19_09,915.bmp',
    'Large': 'C:\\Users\\danie\\Documents\\GitHub\\850\\data\\Data\\Test\\Large\\Crack__20180419_13_29_14,846.bmp'
    
}

#Plot the image with predictions
for true_label, image_path in test_image_paths.items():
    image_name = os.path.basename(image_path)
    print(f"Processing image: {image_name}")  #Print the image name to the console and display image

    probabilities = predict_image_class(model, image_path)[0]
    predicted_label = np.argmax(probabilities)

    #picture set up
    img = image.load_img(image_path)  #Load image without rescaling for display
    plt.imshow(img)
    plt.title(f'{image_name}\nTrue Crack Classification Label: {true_label}') #prints size
    plt.xlabel('\n'.join([f'{label}: {prob:.2%}' for label, prob in zip(class_indices.keys(), probabilities)]))
    
    # Remove the ticks and show the plot
    plt.xticks([])
    plt.yticks([])
    plt.show()
