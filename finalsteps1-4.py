from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
import json
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt

#Define the image input shape
image_shape = (100, 100, 3)  # Width, Height, Channels

#Data augmentation for the training data
train_datagen = ImageDataGenerator(
    rescale=1./255,       # rescaling
    shear_range=0.2,      # shear intensity
    zoom_range=0.2,       # zoom range
    horizontal_flip=True  # randomly flip inputs horizontally
)

#Rescale validation set
validation_datagen = ImageDataGenerator(rescale=1./255)

#Directories for training and validation data
data_folder_path = 'C:\\Users\\danie\\Documents\\GitHub\\850\\data\\Data'
train_data_dir = os.path.join(data_folder_path, 'Train')
validation_data_dir = os.path.join(data_folder_path, 'Validation')

#Creating the train and validation generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_shape[0], image_shape[1]),  # resize image
    batch_size=32,
    shuffle=True, #shuffles images
    class_mode='categorical'  # more than 2 classes thus categorical
)
#Valedation generator
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(image_shape[0], image_shape[1]), # resize image
    batch_size=32,
    shuffle=True, #shuffles images
    class_mode='categorical' # more than 2 classes thus categorical
)

#Calculate class weights to not give importance to large only
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes)
class_weights_dict = dict(enumerate(class_weights))

#Outputting the classes found by the generator
print(train_generator.class_indices)
print(validation_generator.class_indices)

#Model
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(100, 100, 3),activation='relu')) #down from 32, relu being used
model.add(BatchNormalization()) #normalizing the layer
model.add(LeakyReLU(alpha=0.1)) #constant
model.add(MaxPooling2D(pool_size=(2, 2))) #constant
model.add(Dropout(0.25)) #increasing now slowly

model.add(Conv2D(32, (3, 3),activation='relu')) #down from 128 to 32, relu used
model.add(BatchNormalization()) #normalizing the layer
model.add(LeakyReLU(alpha=0.1))#constant
model.add(MaxPooling2D(pool_size=(2, 2))) #constant
model.add(Dropout(0.5))#increasing now slowly

model.add(Flatten()) #flatten layer
model.add(Dense(64, activation='relu')) #reduced neurons from 256 to 128 to 64, still using relu
model.add(Dropout(0.65))#increasing now slowly
model.add(Dense(4, activation='softmax')) #specified 4 neurons, using softmax here

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
model.summary() #learning rate down uyp from 0.000005

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) #early stopping if val loss doesnt change

#Train the model
history = model.fit(
    train_generator,
    epochs=40, #up from 25
    validation_data=validation_generator,
    callbacks=[early_stopping],
    class_weight=class_weights_dict  #using class weight now
)

#Plotting training and validation accuracy values

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#Save the model path using keras
model.save('C:\\Users\\danie\\Documents\\GitHub\\850_2\\my_model1.keras')

#Save class indices with json
class_indices = train_generator.class_indices
with open('C:\\Users\\danie\\Documents\\GitHub\\850_2\\class_indices.json', 'w') as class_file:
    json.dump(class_indices, class_file)
