Self-Driving Car Engineer Nanodegree
Deep Learning
Project: Build a Traffic Sign Recognition Classifier
In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary.
Note: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to \n", "File -> Download as -> HTML (.html). Include the finished document along with this notebook as your submission.
In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a write up template that can be used to guide the writing process. Completing the code template and writeup template will cover all of the rubric points for this project.
The rubric contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
Note: Code and Markdown cells can be executed using the Shift + Enter keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.
Step 0: Load The Data
In [34]:

# Load pickled data
import pickle
​
# TODO: Fill this in based on where you saved the training and testing data
​
training_file = 'traffic-signs-data/train.p'
testing_file = 'traffic-signs-data/test.p'
​
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
print(len(X_train))
​
assert len(test['features']) == len(test['labels'])
assert len(train['features']) == len(train['labels'])
39209
Step 1: Dataset Summary & Exploration
The pickled data is a dictionary with 4 key/value pairs:
'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
'labels' is a 1D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
'sizes' is a list containing tuples, (width, height) representing the the original width and height the image.
'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES
Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the pandas shape method might be useful for calculating some of the summary results.
Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas
In [2]:

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
​
import numpy as np
import pandas as pd
​
# TODO: Number of training examples
n_train = len(train['features'])
​
# TODO: Number of testing examples.
n_test = len(test['features'])
​
# TODO: What's the shape of an traffic sign image?
image_shape = train['features'][0].shape
​
# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(train['labels']))
​
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
​
Number of training examples = 39209
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
Include an exploratory visualization of the dataset
Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.
The Matplotlib examples and gallery pages are a great resource for doing visualizations in Python.
NOTE: It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections.
In [3]:

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import random
​
# Visualizations will be shown in the notebook.
%matplotlib inline
​
index = random.randint(0, len((train['labels'])))
image = train['features'][index].squeeze()
​
plt.figure(figsize=(1,1))
plt.imshow(image)
​
Out[3]:
<matplotlib.image.AxesImage at 0x1234fcba8>

In [4]:

plt.hist(train['labels'],bins=range(len(np.unique(train['labels']))))
Out[4]:
(array([  210.,  2220.,  2250.,  1410.,  1980.,  1860.,   420.,  1440.,
         1410.,  1470.,  2010.,  1320.,  2100.,  2160.,   780.,   630.,
          420.,  1110.,  1200.,   210.,   360.,   330.,   390.,   510.,
          270.,  1500.,   600.,   240.,   540.,   270.,   450.,   780.,
          240.,   689.,   420.,  1200.,   390.,   210.,  2070.,   300.,
          360.,   480.]),
 array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42]),
 <a list of 42 Patch objects>)

Step 2: Design and Test a Model Architecture
Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the German Traffic Sign Dataset.
There are various aspects to consider when thinking about this problem:
Neural network architecture
Play around preprocessing techniques (normalization, rgb to grayscale, etc)
Number of examples per label (some have more than others).
Generate fake data.
Here is an example of a published baseline model on this problem. It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.
NOTE: The LeNet-5 implementation shown in the classroom at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play!
Pre-process the Data Set (normalization, grayscale, etc.)
Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.
In [5]:

##TODO: normalize frequency of the images 
##TODO: create jittered dataset - shift position and rotate angle / transform, etc
##TODO: experiment with hue, contrast, etc. 
##TODO: experiment with adjusting greyscale
​
def prep_img(img):
​
    return img / 255 * 0.8 + 0.1
In [6]:

##Image before processing
index = 333
image_pre = X_train[index]
In [35]:

### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.
​
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
​
​
processed_train = np.array([prep_img(X_train[i]) for i in range(len(X_train))], dtype = np.float32)
processed_test = np.array([prep_img(X_test[i]) for i in range(len(X_test))], dtype = np.float32)
X_test = processed_test
​
print(len(processed_train))
print(len(X_train))
print(len(train['labels']))
​
image_post = processed_train[index]
​
​
##Split and Shuffle the images
​
X_train, X_validation, y_train, y_validation = train_test_split(processed_train,train['labels'])
​
print(len(X_train))
print(len(X_validation))
​
39209
39209
39209
29406
9803
In [8]:

#index = random.randint(0, len(X_train))
plt.figure(figsize=(3,3))
plt.imshow(image_pre, cmap="gray")
​
Out[8]:
<matplotlib.image.AxesImage at 0x125d65198>

In [9]:

post_proc = prep_img(image_pre)
plt.figure(figsize=(3,3))
plt.imshow(post_proc)
Out[9]:
<matplotlib.image.AxesImage at 0x125e10518>

Model Architecture
In [10]:

##Set up tensorflow and main hyperparameners
​
import tensorflow as tf
from tensorflow.contrib.layers import flatten
​
​
EPOCHS = 15
BATCH_SIZE = 128
learning_rate = 0.0005
In [11]:

### Define your architecture here.
### Feel free to use as many code cells as needed.
​
def NetLaNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
​
    # Activation functions. Went for ReLu
    conv1 = tf.nn.relu(conv1)
​
    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
​
    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation.
    conv2 = tf.nn.relu(conv2)
​
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
​
    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Activation.
    fc1 = tf.nn.relu(fc1)
    
    # Dropout - to prevent overfitting. 
    fc1 = tf.nn.dropout(fc1, keep_prob)
​
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation.
    fc2    = tf.nn.relu(fc2)
    
    # Dropout - to prevent overfitting. 
    fc2 = tf.nn.dropout(fc2, keep_prob)
​
    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits
​
​
Train, Validate and Test the Model
A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation sets imply underfitting. A high accuracy on the test set but low accuracy on the validation set implies overfitting.
In [12]:

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
​
​
​
In [13]:

##ONE-HOT-ENCODE and PREP VARIABLES
​
​
from datetime import datetime
​
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
keep_prob = tf.placeholder(tf.float32)
In [14]:

##TRAIN PIPELINE
​
logits = NetLaNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss_operation)
In [15]:

##EVALUATE PIPELINE
​
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
​
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob : 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
In [16]:

##TRAINING THE MODEL
​
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob : 0.6})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    filename = './net/netlanet_03'
    saver.save(sess, filename)
    print("Model saved")
Training...

EPOCH 1 ...
Validation Accuracy = 0.399

EPOCH 2 ...
Validation Accuracy = 0.562

EPOCH 3 ...
Validation Accuracy = 0.726

EPOCH 4 ...
Validation Accuracy = 0.810

EPOCH 5 ...
Validation Accuracy = 0.856

EPOCH 6 ...
Validation Accuracy = 0.884

EPOCH 7 ...
Validation Accuracy = 0.898

EPOCH 8 ...
Validation Accuracy = 0.924

EPOCH 9 ...
Validation Accuracy = 0.932

EPOCH 10 ...
Validation Accuracy = 0.938

EPOCH 11 ...
Validation Accuracy = 0.942

EPOCH 12 ...
Validation Accuracy = 0.950

EPOCH 13 ...
Validation Accuracy = 0.955

EPOCH 14 ...
Validation Accuracy = 0.960

EPOCH 15 ...
Validation Accuracy = 0.962

Model saved
In [17]:

#Launch the model on the test data
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./net'))
​
    test_accuracy = sess.run(accuracy_operation, feed_dict={x: X_test, y: y_test, keep_prob : 1.0})
​
print('Test Accuracy: {}'.format(test_accuracy))
Test Accuracy: 0.9008709192276001
Step 3: Test a Model on New Images
To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
You may find signnames.csv useful as it contains mappings from the class id (integer) to the actual sign name.
Load and Output the Images
In [18]:

### Load the images and plot them here.
### Feel free to use as many code cells as needed.
In [19]:

##Helper to change the loaded images to RGB scheme
​
import cv2
​
def rgb(img):
    b,g,r = cv2.split(img)
    img2 = cv2.merge([r,g,b])
    return img2
    
    
In [20]:

image1 = rgb(cv2.imread('./unseen/01-noentry.jpg'))
image2 = rgb(cv2.imread('./unseen/02-way.jpg'))
image3 = rgb(cv2.imread('./unseen/03-35tons.jpg'))
image4 = rgb(cv2.imread('./unseen/04-giveway.jpg'))
image5 = rgb(cv2.imread('./unseen/05-slipperyroad.jpg'))
In [21]:

plt.imshow(image1)
plt.show()

In [22]:

plt.imshow(image2)
plt.show()

In [23]:

plt.imshow(image3)
plt.show()

Predict the Sign Type for Each Image
In [24]:

plt.imshow(image4)
plt.show()

In [25]:

plt.imshow(image5)
plt.show()

In [26]:

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
​
​
image = cv2.resize(image1,(32,32))
image = prep_img(image)
​
plt.imshow(image)
​
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./net'))
    
    new_img = np.expand_dims(image,axis=0)
    
    predict = tf.argmax(logits, 1)
    probabilities = logits
    
    
    print("Prediction: ", predict.eval(feed_dict = {x : new_img, keep_prob : 1.0 }))
    print("Top 5: ", sess.run(tf.nn.top_k(tf.nn.softmax(logits.eval(feed_dict = {x : new_img, keep_prob : 1.0 })),5)))
Prediction:  [15]
Top 5:  TopKV2(values=array([[  9.99998331e-01,   1.18835203e-06,   4.15543866e-07,
          6.32811918e-08,   1.77183495e-08]], dtype=float32), indices=array([[15,  9,  8,  2, 13]], dtype=int32))

In [27]:

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
​
​
image = cv2.resize(image2,(32,32))
image = prep_img(image)
​
plt.imshow(image)
​
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./net'))
    
    new_img = np.expand_dims(image,axis=0)
    
    predict = tf.argmax(logits, 1)
    probabilities = logits
    
    
    print("Prediction: ", predict.eval(feed_dict = {x : new_img, keep_prob : 1.0 }))
    print("Top 5: ", sess.run(tf.nn.top_k(tf.nn.softmax(logits.eval(feed_dict = {x : new_img, keep_prob : 1.0 })),5)))
Prediction:  [38]
Top 5:  TopKV2(values=array([[  9.99996781e-01,   3.06336483e-06,   8.07773617e-08,
          2.10219925e-10,   1.86920163e-12]], dtype=float32), indices=array([[38, 40, 34, 20, 36]], dtype=int32))

In [28]:

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
​
​
image = cv2.resize(image3,(32,32))
image = prep_img(image)
​
plt.imshow(image)
​
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./net'))
    
    new_img = np.expand_dims(image,axis=0)
    
    predict = tf.argmax(logits, 1)
    probabilities = logits
    
    
    print("Prediction: ", predict.eval(feed_dict = {x : new_img, keep_prob : 1.0 }))
    print("Top 5: ", sess.run(tf.nn.top_k(tf.nn.softmax(logits.eval(feed_dict = {x : new_img, keep_prob : 1.0 })),5)))
Prediction:  [2]
Top 5:  TopKV2(values=array([[ 0.63844633,  0.1764199 ,  0.09371448,  0.04396971,  0.01673588]], dtype=float32), indices=array([[2, 1, 7, 5, 8]], dtype=int32))

In [29]:

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
​
​
image = cv2.resize(image4,(32,32))
image = prep_img(image)
​
plt.imshow(image)
​
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./net'))
    
    new_img = np.expand_dims(image,axis=0)
    
    predict = tf.argmax(logits, 1)
    probabilities = logits
    
    
    print("Prediction: ", predict.eval(feed_dict = {x : new_img, keep_prob : 1.0 }))
    print("Top 5: ", sess.run(tf.nn.top_k(tf.nn.softmax(logits.eval(feed_dict = {x : new_img, keep_prob : 1.0 })),5)))
Prediction:  [13]
Top 5:  TopKV2(values=array([[  9.99994516e-01,   4.95367203e-06,   5.09160259e-07,
          6.97251146e-10,   2.05625267e-10]], dtype=float32), indices=array([[13,  9, 10, 12,  3]], dtype=int32))

In [30]:

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
​
​
image = cv2.resize(image5,(32,32))
image = prep_img(image)
​
plt.imshow(image)
​
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./net'))
    
    new_img = np.expand_dims(image,axis=0)
    
    predict = tf.argmax(logits, 1)
    probabilities = logits
    
    
    print("Prediction: ", predict.eval(feed_dict = {x : new_img, keep_prob : 1.0 }))
    print("Top 5: ", sess.run(tf.nn.top_k(tf.nn.softmax(logits.eval(feed_dict = {x : new_img, keep_prob : 1.0 })),5)))
Prediction:  [22]
Top 5:  TopKV2(values=array([[ 0.92606831,  0.03940045,  0.01805967,  0.00679721,  0.00274941]], dtype=float32), indices=array([[22, 29, 25, 26, 31]], dtype=int32))

Analyze Performance
Analyze Performance
Calculate the accuracy for these 5 new images.
For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
Total images = 5 Correct predictions = 3 Accuracy = 60%
Output Top 5 Softmax Probabilities For Each Image Found on the Web
For each of the new images, print out the model's softmax probabilities to show the certainty of the model's predictions (limit the output to the top 5 probabilities for each image). tf.nn.top_k could prove helpful here.
The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
tf.nn.top_k will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. tk.nn.top_k is used to choose the three classes with the highest probability:
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
Running it through sess.run(tf.nn.top_k(tf.constant(a), k=3)) produces:
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
Looking just at the first row we get [ 0.34763842,  0.24879643,  0.12789202], you can confirm these are the 3 largest probabilities in a. You'll also notice [3, 0, 5] are the corresponding indices.
In [31]:

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
​
### Top 5 softmax are included below the images in their respective cells above
Note: Once you have completed all of the code implementations, you need to finalize your work by exporting the IPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run. You can then export the notebook by using the menu above and navigating to \n", "File -> Download as -> HTML (.html). Include the finished document along with this notebook as your submission.
Project Writeup
Once you have completed the code implementation, document your results in a project writeup using this template as a guide. The writeup can be in a markdown or pdf file.