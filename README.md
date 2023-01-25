# Project Information

Recognizing traffic signs with CNN

## Information About the Data Set

The dataset we use is the GTSDB dataset, an accessible dataset that is widely used for the evaluation and testing of Traffic Sign Detection and Traffic Sign Classification systems. Dataset Houben et al. (2013) prepared by. Traffic signs belong to 43 different classes. There are close to 50 thousand photos.

## Model Building

### Used Media/Language, Libraries and Versions

#### Libraries

    Numpy, Pandas, Os, Keras, PIL, tensorflow, matplotlib,

#### IDE

 Jupyter Notebook

#### Method

CNN

# Project development steps

First, we imported our libraries into our working environment. We have imported the necessary libraries into our working environment to use in the recognition of traffic signs.
Then we imported the dataset into the working environment. In this data set, we divided the data into training and test sets. Since images are read in BGR format by default in the OpenCV library, we converted the images to RGB format. Then we adjusted the min-max normalization.

``` python

import numpy as np  
import pandas as pd 
import os  
import cv2 

from PIL import Image 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

from sklearn.model_selection import train_test_split #Training set - Validation set split
from skimage.transform import resize
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Conv2D, MaxPool2D, AveragePooling2D, Dense, Activation, Flatten, Dropout
from keras.optimizers import Adam, SGD
from keras.regularizers import L1, L2

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
```

Then it will display how many classes are in it.We were pissed. Finally, we performed a normalization data check.

``` python

data = [] 
labels = []
train_path = 'dataset/archive/Train'
classes = len(os.listdir(train_path))

for i in os.listdir(train_path):
    dir = train_path + '/' + i
    for j in os.listdir(dir):
        img_path = dir+'/'+j
        # Reading images
        img = cv2.imread(img_path,-1)
        # OpenCV default color is BGR. So, we are converting it to RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resizing the images
        img = cv2.resize(img, (30,30), interpolation = cv2.INTER_NEAREST)
        # Normalizing the images (Min-Max Normalization)
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        data.append(img)
        labels.append(i)
        
data = np.array(data)
labels = np.array(labels)
```

We divided our data set as 80% training and 20% validation set. We performed the training of the model and the hyperparameter adjustment operations on the training set. We performed One-Hot Encoding to represent categorical variables in binary.

## Splitting Training / Validation Set

``` python
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size= 0.20, random_state=42)

print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_val shape  : ', X_val.shape)
print('y_val shape  : ', y_val.shape)

X_train shape:  (31367, 30, 30, 3)
y_train shape:  (31367,)
X_val shape  :  (7842, 30, 30, 3)
y_val shape  :  (7842,)
```

## Converting labels into One-Hot Encoding

``` python
y_train = to_categorical(y_train, classes)
y_val = to_categorical(y_val, classes)

y_train

array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 1., 0., ..., 0., 0., 0.],
       [0., 0., 1., ..., 0., 0., 0.]], 
       dtype=float32)
       
```

After preparing and preprocessing the image for creating the first CNN model, we set the number of filters to 16 and the kernel size to (5,5) in the Convolution layer. In the first Convolution layer, the width and height values of the image must be entered. The image size will be adjusted and feature extraction will be performed according to these values. The activation function used is already ReLU. In the Pooling layer, we reduced the data by creating a (2,2) pool size. In the output section, we used the Softmax function to classify and obtained the probabilities. The result of the classification is the class with the highest probability. Optimizer = 'ADAM', lr (learning rate) =0.0001, batch_size variable is 64, and the epoch is set to 20.

## Building CNN Models

### Baseline Model

``` python
model = Sequential()

# Convolutional Step 1
model.add(Conv2D(input_shape=(30,30,3), filters=16, kernel_size=(5,5), padding="same", activation="relu"))

# Max Pooling Step 1
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# Convolutional Step 2
model.add(Conv2D(filters=16, kernel_size=(5,5), padding="same", activation="relu"))

# Max Pooling Step 2
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Flatten Step : matris değerlerini alt alta yazar
model.add(Flatten())

############################
#  FULLY CONNECTED LAYERS  #
############################

# Fully Connected Layer 1
model.add(Dense(units=128,activation="relu", kernel_initializer='he_uniform'))

# Fully Connected Layer 2
model.add(Dense(units=43, activation="softmax")) 

print("MODEL SUMMARY\n")

model.summary()
```

In Artificial Neural Networks, as you know, there is a cost calculation. In this calculation, we want the cost to be minimum. Therefore, calculating the loss value is very important for us. Then, the data was processed, and even at the beginning of the model, we calculated an accuracy value of 94.6%. We plotted the change in loss values and accuracy values on a graph.

image.png

image.png

* ### Model 1

Then, without changing other hyperparameters, we doubled the number of filters and set it to 32. We examined the effect of this on the model. In this case, we achieved an accuracy rate of 96.5%.

``` python
model = Sequential()

# Convolutional Step 1
model.add(Conv2D(input_shape=(30,30,3), filters=32, kernel_size=(5,5), padding="same", activation="relu"))

# Max Pooling Step 1
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# Convolutional Step 2
model.add(Conv2D(filters=32, kernel_size=(5,5), padding="same", activation="relu"))

# Max Pooling Step 2
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Flatten Step
model.add(Flatten())

############################
#  FULLY CONNECTED LAYERS  #
############################

# Fully Connected Layer 1
model.add(Dense(units=128,activation="relu", kernel_initializer='he_uniform'))

# Fully Connected Layer 2
model.add(Dense(units=43, activation="softmax")) 

print("MODEL SUMMARY\n")

model.summary() 
```

We used the Softmax function to determine which class the output from the model could belong to. The neuron with the highest probability is determined as the class of the image. After these operations, we printed the confusion matrix on the screen.

image.png

* ### Model 1 with SGD Optimizer

We wanted to observe the effects of the selected optimization algorithm on the model. The selected optimization method may give good results in some problems and bad results in others. To observe the effect of the optimization method on our model, we set the optimization method to Stochastic Gradient Descent (SGD) without changing other hyperparameters.

``` python
model = Sequential()

# Convolutional Step 1
model.add(Conv2D(input_shape=(30,30,3), filters=32, kernel_size=(5,5), padding="same", activation="relu"))

# Max Pooling Step 1
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# Convolutional Step 2
model.add(Conv2D(filters=32, kernel_size=(5,5), padding="same", activation="relu"))

# Max Pooling Step 2
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Flatten Step
model.add(Flatten())

############################
#  FULLY CONNECTED LAYERS  #
############################

# Fully Connected Layer 1
model.add(Dense(units=128,activation="relu", kernel_initializer='he_uniform'))

# Fully Connected Layer 2
model.add(Dense(units=43, activation="softmax")) 

print("MODEL SUMMARY\n")

model.summary()
```

In this case, the accuracy value we obtained was 73.986%. Stochastic Gradient Descent (SGD) gave a lower accuracy than 'ADAM'

``` python
best_val_accuracy = max(history.history['val_accuracy'])
print("Best Validation Accuracy: %",best_val_accuracy*100)
```

Best Validation Accuracy: % 73.98622632026672

Since we received a lower accuracy with SGD, we continued to use ADAM as the optimization method. However, this time we reduced the size of the filters used in the Convolutional layers from (5x5) to (3x3). By using smaller filters, we aimed to extract more information from the images. In this case, the accuracy we obtained was 97.13%.

``` python
best_val_accuracy = max(history.history['val_accuracy'])
print("Best Validation Accuracy: %",best_val_accuracy*100)
```

Best Validation Accuracy: % 97.13083505630493

* ### Model 3

When we came to Model 3, we made our model a little deeper by adding 2 more convolutional layers as an extra. The result was %97.9.

``` python
model = Sequential()

# Convolutional Step 1
model.add(Conv2D(input_shape=(30,30,3), filters=32, kernel_size=(3,3), padding="same", activation="relu"))

# Convolutional Step 2
model.add(Conv2D(input_shape=(30,30,3), filters=32, kernel_size=(3,3), padding="same", activation="relu"))

# Max Pooling Step 1
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# Convolutional Step 3
model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))

# Convolutional Step 4
model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))

# Max Pooling Step 2
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Flatten Step
model.add(Flatten())

############################
#  FULLY CONNECTED LAYERS  #
############################

# Fully Connected Layer 1
model.add(Dense(units=128,activation="relu", kernel_initializer='he_uniform'))

# Fully Connected Layer 2
model.add(Dense(units=43, activation="softmax")) 

print("MODEL SUMMARY\n")

model.summary()
````

We also decided to do Batch normalization. This method is used to make evolutionary neural networks regular. This can also reduce the training time and provide better performance of the model. The result showed that this is a correct possibility and we got %99.1 accuracy.
During the training to prevent overfitting, we used drop-out to forget some neurons. Our network was large enough and data count was low, so it carried a risk of overfitting. We wanted to prevent this. Initially, we defined Drop-out (0.3) and the result was %99.5.

``` python
model = Sequential()

# Convolutional Step 1
model.add(Conv2D(input_shape=(30,30,3), filters=32, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())

# Convolutional Step 2
model.add(Conv2D(input_shape=(30,30,3), filters=32, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())

# Max Pooling Step 1
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# Convolutional Step 3
model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())

# Convolutional Step 4
model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())

# Max Pooling Step 2
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Flatten Step
model.add(Flatten())
model.add(Dropout(0.3))

############################
#  FULLY CONNECTED LAYERS  #
############################

# Fully Connected Layer 1
model.add(Dense(units=256,activation="relu", kernel_initializer='he_uniform'))
model.add(Dropout(0.2))

# Fully Connected Layer 2
model.add(Dense(units=128,activation="relu", kernel_initializer='he_uniform'))
model.add(Dropout(0.2))

# Fully Connected Layer 3
model.add(Dense(units=43, activation="softmax")) 

print("MODEL SUMMARY\n")

model.summary()

```

We increased the drop-out ratio a bit more (0.5) and this time we saw a very high value like %99.92. From this stage, we decided to try the trained model on unseen images in the test set.
We had 12630 photos in the test set. First, we got the names and labels of the images from the 'csv' file where the image information is in the test set. We performed image preprocessing in the path where the images are located. We converted the images from BGR format to RGB format. We re-sized them to 30x30 and performed min-max normalization on the image pixels.

``` python
y_test = labels

#One-Hot Encoding with Keras
y_test = to_categorical(y_test)

model.evaluate(X_test, y_test)
```

395/395 [==============================] - 4s 9ms/step - loss: 0.1271 - accuracy: 0.9659
[0.1271226853132248, 0.9658749103546143]

When we tested the trained model on the test set, the model obtained results that were about 96.5% accurate.

To visually observe the results on the images, we examined some images in the dataset for the predicted labels and the actual labels. We printed the correctly predicted labels in green and the incorrectly predicted ones in red on the screen.
image.p

# Conclusion

In our model, we obtained an accuracy value of 99% in the validation set and a high result of 96.58% in another test set. Therefore, we believe that our model is successful. It is also a project that has found its way into our daily lives. Traffic sign detection and recognition is an important technological element in advanced driving support systems. The areas of use of this feature include autonomous driving, advanced driving support systems, creating and maintaining traffic sign maps, etc. This is a challenging real-world computer vision problem due to the difficulties caused by lighting, the fact that the front of the sign may be covered or obscured by other objects such as vehicles or trees, changes in the viewing angle, weather or lighting conditions, fading on the signs, and variations caused by human or production-related factors on the signs. Therefore, the data set is also very important. We also used one of the most widely used data sets in the world.
