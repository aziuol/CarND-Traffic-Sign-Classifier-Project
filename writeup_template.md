# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model rchitecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

[image9]: ./examples/labels_histogram.png "Training Data - Classifier Label Histogram"
[image10]: ./examples/BoxPlot_TrainingData.png "Training Data - Box Plot"
[image11]:
[blue_text](http://benchmark.ini.rub.de/?section=gtsrb&subsection=about)
[image12]: ./examples/preprocessing_random_30_classes.png "Pre-processing Random Class visualisation"
[image13]: ./examples/LeNet_GTSRB.png "LeNet GTSRB Covnet"
[image14]: ./examples/accuracy_loss.png "Accuracy and loss of implemented Covnet architecture"
[image15]: ./examples/custom.png "New Web Images"
[image16 ]: ./examples/Visualisation_of_layers.png "1st and 2nd covnet layers"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library for a basic summary of the Data set, and then used the pandas library to calculate summary statistics of the traffic signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of test examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43
* Number of class names = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
![Traffic Sign Training Data Historgram][image9]

From the histogram representation of the training data set we observe that distribution of the sample data is skwewed. 
The median class size is 540 and the 75 Percentile size is 1275.
23 signs sit equal and below the median sample size.  

![Box Plot of Training Data set][image10] 
![Traffic Sign - class images training data][image11] 

The distribution of the validation dataset is the similar to that of the training dataset.
As described by [The Institute for Fuer Neruroinformatik ](http://benchmark.ini.rub.de/?section=gtsrb&subsection=about) the dataset itself is large and with unbalanced frequencies, with large variations in visual appearances due to lighting conditions, partial occlusions, angle of view etc.  Within the datasets there are also subsets of classes such as speed limit signs and pictograms that are similar to each other  that may prove difficult for the Covnet to classify accurately. I shall discuss these aspects further down in the discussion.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As part of the preprocessing, all samples were converted to grayscale.  This was implemented as it was stated by .[LeCun and Sermanet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) that this single process improved the accuracy of their state of the art  Covnet from 98.97% to 99.17%.

The second step of the preprocessing was to approximately normalise the image using `(pixel - 128)/ 128` so that the data has a mean of zero and equal variance.  ![Visualisation of 30 classes post pre-processing is illustrated below ][image12]

Additional discussion point: I spent quite a bit of time augmenting the data. The idea was to apply a random affine augmentation that would closely simulate the application - the classifier should be able to distinguish trafic signs observered at different camera angles or perspectives.  In addition, the parameter variation also allowed me to easily scale up my datasets. I chose to scale up and balance to the 90th percentile.

Unfortunately employing the extended augmented dataset as described above, the accuracy of teh covnet on new sign images proved challenging and covnet actuall performed worse  despite fast convergence with low epoch and very high accuracy on the validation set. I will need to investigate this further - I have thus not included the above in my discussion or notebook.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model was implemented using the same LeNet architecture discused in the lecture notes using TensorFlow. ![LeNet Covnet][image13]

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters:  
`epochs = 50
batch_size = 128
one_hot_y = tf.one_hot(y, 43)
rate = 0.001
logits, conv1, conv2 = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)`

I used teh Adam (Adaptive Moment Esitamation) Optimisor as studies have shown it to perform very well relative to other optimisos especially in it's abilty to quickly converge as described in this article [here](http://ruder.io/optimizing-gradient-descent/index.html).



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of `Validation Accuracy = 94.603%`
* test set accuracy of `Test Accuracy = 92.732%`

If an iterative approach was chosen:

* Initially my architecture performed poorly becuase I did not adjust my output layers -> this was a quick fix!
* I also initially included a training split, and also drop-pout on my 4th layer.  This was performing well according with the validation set and testing set.  It was also clear when the network was overfitting as I could see it bouncing back and forth on it's accuracy.
* My current architecture is a good starting point, I should attempt to improve it without comprimising the certainty and accuracy of the predictions one way to do so would be to add drop-out before my final fully connected layer  and also try skipping some layers.  This was mentioned as possible techniques in the lecture material. Also I need to determine why my original architecture which included affine transforms for extending classes did not perform well with new images despite having a high validation and testing accuracy.
![Below is the accuracy and loss performance of teh architecture][image14]  by slowing down the learning rate, the loss would smooth out, it appears that the learning rate is a little too coarse.  By adding more epochs this will smooth out.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
![shown in addition with their pre-processing steps][image15]

The fourth image may be difficult as the background also includes segments of other signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The accuracy of the model's predcitions was 100%.  This faired very well in comparison to my first architecture attempt where the covnet was deeply confused with the new images!

`Accuracy for custom sign = 100.000%`
Here are the results of the prediction using the top 5 softmaxes.

`
No.1 
[ 1  7 13 25 36]
predicted sign names 
['Speed limit (30km/h)', 'Speed limit (100km/h)', 'Yield', 'Road work', 'Go straight or right']

No.2 
[ 4  8 35 20 25]
predicted sign names 
['Speed limit (70km/h)', 'Speed limit (120km/h)', 'Ahead only', 'Dangerous curve to the right', 'Road work']

No.3 
[ 5  5 26 22 38]
predicted sign names 
['Speed limit (80km/h)', 'Speed limit (80km/h)', 'Traffic signals', 'Bumpy road', 'Keep right']

No.4 
[ 3  2 33 23 18]
predicted sign names 
['Speed limit (60km/h)', 'Speed limit (50km/h)', 'Turn right ahead', 'Slippery road', 'General caution']

No.5 
[31 40 34 24  1]
predicted sign names 
['Wild animals crossing', 'Roundabout mandatory', 'Turn left ahead', 'Road narrows on the right', 'Speed limit (30km/h)']


actual labels 
[ 1  7 13 25 36]
actual sign names 
['Speed limit (30km/h)', 'Speed limit (100km/h)', 'Yield', 'Road work', 'Go straight or right']
`
The model was able to correctly guess all 5 new traffic signs.  I will need to continue testing it with new traffic sign images to cover all the classes, and opt to include or catogorise "challenge" signs.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

`[[0.998 0.002 0.000 0.000 0.000]
 [0.999 0.001 0.000 0.000 0.000]
 [1.000 0.000 0.000 0.000 0.000]
 [1.000 0.000 0.000 0.000 0.000]
 [0.999 0.001 0.000 0.000 0.000]]`


For the first sign "Speed limit (30km/h)" it is 99.8% certain of it's prediction.  The next prediction is 'Speed limit (100km/h)'
For the second sign "Speed limit (70km/h)" it is 99.9% certain of it's prediction.  The next prediction is 'Speed limit (120km/h)'
For the third sign "Speed limit (80km/h)" it is 100% certain of it's prediction.
For the fourth sign "Speed limit (60km/h)" it is 100% certain of it's prediction.
For the fifth sign  "Wild animals crossing" it is 99.9% certain with the next prediction being  'Roundabout mandatory'

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
![Visualisation of first and second layers from the covnet][image16]

I used 'Speed limit (30km/h)' to evaluate feature maps of the neural network.  Interestingy it appears to be using the inverse segments for detection - it's classifiying the blobs around the dark digits to classify, that is "light field" classification.  I should check this against other signs to see if this is a correct analogy.
