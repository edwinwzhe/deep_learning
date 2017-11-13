---
title:  "Machine Learning Engineer Nanodegree"
date:   2017-10-30 20:00:00
categories: [deep learning]
tags: [python, keras, machine learning, CNN, transfer learning]
---

## Capstone Project

Edwin He <br/>
November 7th, 2017

## __I. Definition__

## Project Overview
Deep learning with Deep Neural Network has gained huge momentum in the past few years owning to the blooming of data and GPU accelerated computation. The former makes it possible to collect massive amount of training data to tune millions or even billions of parameters; the later makes running deep learning computation approachable for everyone have access to a consumer PC with GPU installed.

One of the most exciting fields of deep learning application is computer vision which gives machine the ability to see. One of the most influential innovations in the field of computer vision is Convolutional Neural Network(CNN), first published by LeCun et al [1], became prominence as Alex Krizhevsky used it to win 2012 ImageNet competition. 

In this study, I will train an image classifier using the KTH-ANIMALS dataset[4] to classify 19 different animals with convolution neural network(CNN). Also look to utilize the trained weight for object localization.

## Problem Statement

Though empowered by GPU, training an image classifier with CNN still requires a long computation time. CNN needs to go deeper to achieve good result on more complex tasks which requires massive amount of training data for tuning the weights. Neither is often feasible for individuals. 

If we imagine how we learn as a human, we do not learn to recognize every object from stretch. After learning how a bike looks like and how it works, even for the first time we see a car, we can immediate transfer our knowledge to bikes and we can tell immediately a car is most likely a bike with more wheels and bigger in size. 

Similarly after training a network to learn dogs and cats, we should not have to learn to classify a wolf and tiger from stretch because we have already learned how the face and body look like for dog or cat like animals.

There are many CNN models already trained with ImageNet (a massive image data set), i.e. it has learned to classify the objects in the ImageNet and have also learned the parts that assembles them. I will transfer the knowledge learned from classifying ImageNet objects to our tasks without having to learn from stretch. This will solve our problem as it greatly reduces the number of trainable weights in our model which requires way less data and effort to learn. 


## Metrics

Classification Accuracy, Cohen's kappa and Cross-Entropy Loss are used as evaluation metrics. 

* Classification Accuracy is simply the percentage of correctly classified images, as the dataset is slightly imbalanced. This metric will be for reference only.

* Cohen's kappa is a score that expresses the level of agreement between two annotators, ranging from -1 to 1 with 1 being perfectly agreement. It is a more accurate performance measurement for model with imbalanced dataset. Formula 1 defines Cohen's kappa.
![Kappa Formula]({{ site.url }}/assets/mlnano_capstone/kappa_formula.png) <br/>
_Formula 1: Kappa Formula_

* Cross-Entropy Loss is a metric used with softmax function which generates probabilities for all outcomes. It will give me more information about how confident the model is for correct predictions as well as how far from correct the incorrect predictions are. Cross-Entropy Loss is also used as a metric to learn whether the model is improving or not for early stopping and model checkpoints. Formula 2 defines it.<br/>
![LogLoss Formula]({{ site.url }}/assets/mlnano_capstone/logloss_formula.png) <br/>
_Formula 2: Cross-Entropy Loss [6]_


## __II. Analysis__

## Data Exploration

The KTH-ANIMALS dataset[3] contains only 1740 images in total and are of different shapes from ~200 to ~400 pixels width and height. 
Figure 1 shows the images in their original shape. 

![Input Images]({{ site.url }}/assets/mlnano_capstone/input_images_original.png)
_Figure 1: Images loaded as 224x224_

## Exploratory Visualization

Figure 2 shows the KDE plot of R,G,B channels for each class. It shows the color distribution for different class. I can see that animals in dark yellow such as coyote, horse, leopard, lion and tiger contributes to the peaks on the left of the red channel. Also in general, these pictures are taken in the wild, which explains the significant amount of dark green and dark blue in every most classes coming from the grass field and the sky. 

![KDE By Class]({{ site.url }}/assets/mlnano_capstone/kde_by_class.png)
_Figure 2: KDE plot by class_


Figure 3 shows the label count. The dataset is quite evenly distributed.
![Input Images]({{ site.url }}/assets/mlnano_capstone/training_label_count.png)
_Figure 3: Training Lable Count per Class_

## Algorithms and Techniques

I use Convolutional Neural Network(CNN) to achieve this image classification task. CNN typically includes a few blocks of layers - each includes one of more Convolutional layers followed by Activation layer, Pooling layer and Dropout layer. Finally connecting to one or more fully connected(FC) layers for class prediction.

* Convolutional layer use n-by-n matrics called filters or kernels to slide through the input image performing matrix multiplication. The output becomes the input for the next layer. Figure 3 shows how the output is calculated from a 3x4 input, slided through by a 2x2 kernel with stride 1 (move 1 pixel right/down at a time). 
![ALgo Convolution]({{ site.url }}/assets/mlnano_capstone/algo_convolution.png) <br/>
_Figure 4 Convolution_ [12]

* Activation layer is the layer that introduces non-linearity. ReLU (Recitfied Linear Unit), introduced by Hahnloser et al[11] which simply output the same value when it is greater than 0 and output 0 when it is smaller than 0, is the most popular activation function used in computer vision as it is simple yet effectively solve the vanishing gradient problem in deep neural networks.

* Pooling layer mainly serve as a mean of downsampling to reduce the size of the network. In this study, I use Max Pooling which outputs the max value of the filter. Global Average Pooling is used before the final FC layer. It turns the entire feature map (such as the output in Figure 3) into a single value by averaging the numbers.

* Dropout, raised by Hinton et al[10], is a process that randomly switch off the output of some feature maps forcing the network to learn redundant features which is proven a very good approach to make the model more robust to variations.

The first CNN layer extracts simple patters such as edges. Following layers extract more abstract patterns from the previous layers [4]. 

![Visualize Layers]({{ site.url }}/assets/mlnano_capstone/visualize_layers.png) 
_Figure 5: Visualize Layers_

The first few layers extract relatively simple patterns which are roughly the same for different image classification tasks. For instance, basic patterns like edges and curves are the basic elements of every image [2] (Lee et al 2009). 

![Learning Objects Parts]({{ site.url }}/assets/mlnano_capstone/learning_object_parts.png)
_Figure 6: What Convolutional layers learned_

By reusing the less abstract and less task oriented layers pre-trained by massive data sets, we transfer the knowledge learned from a different tasks to a new task. Transfer learning allows us to reuse the less abstract layers of a pre-trained model to achieve a different goal by training the more abstract/top layers with way less data and effort due to the massive reduction of trainable parameters. 

## Benchmark
The benchmark model - a CNN model with 5 Convolutional layers with similar architecture to VGG19 except that every block has only one CNN layer and Global Average Pooling is used instead of Fully Connected layer to reduce overfitting. 

![Benchmark Model]({{ site.url }}/assets/mlnano_capstone/benchmark_model.png)

A simpler model tend to easily overfit(or underfit if too simple) while a deeper model cannot learn due to the limitation of the insufficient number of training data. 


## __III. Methodology__

## Data Preprocessing
I randomly splitting the data into training, validation and test set containing 1460, 180 and 100 images respectively. Leaving us merely 78 images on average per class which is far from sufficient to train a CNN model from stretch to classify such a diverse dataset. Add another 40 images from Google to the test set (140 images in total).

Images are loaded as 224x224 pixels images. Figure 7 shows the images loaded. 
![Input Images]({{ site.url }}/assets/mlnano_capstone/input_images_224x224.png)
_Figure 7: Images loaded as 224x224_

All image pixels are scaled by 1/255 when being loaded into 4D tensor. 

Image Augmentation is applied to training data to combat overfitting and to make the models more robust. I apply max 10 degrees image rotation, max 0.2 shear and zoom and chances of horizontal flip. 

``` python
datagen = ImageDataGenerator(
    rotation_range=10,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
```
_Code Snippet 1: Image Augmentation_

Figure 8 shows the augmented images. 
![Input Images]({{ site.url }}/assets/mlnano_capstone/input_images_224x224_augmented.png)
_Figure 8 Images Augmented_

## Implementation
The transfer learning model is a VGG19 with weights pre-trained on ImageNet dataset with top layers(two Fully Connected layers) replaced with a top model.

![VGG19]({{ site.url }}/assets/mlnano_capstone/VGG19.png)
_Figure 9: VGG19_


Top Model in Figure 10 is used to replace the two FC layers and the output layer (in purple in Figure 9).
0.7 dropout rate is used in Dropout layer.

![Top Model]({{ site.url }}/assets/mlnano_capstone/top_model.png)
_Figure 10: Top Model_ 


To train the top model in Figure 10, we need to pass the training, validation and test dataset through the VGG19(without top layers included) to obtain the input features (bottleneck features) for the top model. 
``` python
# Load pre-trained VGG19
pretrained_model = applications.VGG19(include_top=False, weights='imagenet')

# Generator for augmented training data 
train_X_generator = datagen.flow(X_train, batch_size=batch_size, shuffle=False)

# Generate bottleneck features
bottleneck_features_train = pretrained_model.predict_generator(train_X_generator, steps=len(X_train) / batch_size)
bottleneck_features_valid = pretrained_model.predict(X_valid)
bottleneck_features_test = pretrained_model.predict(X_test)
```
_Code Snippet 2: Get bottleneck features_

I then use the bottleneck features as the input to train the weight of the top model. 

## Refinement
After training the weights of the top model, I connect the top model trained earlier to the pre-trained VGG19(with top layers removed) to build a new model. 

![Combined Model]({{ site.url }}/assets/mlnano_capstone/combined_model.png)
_Figure 11: final model_

I freeze the first 4 CNN blocks of VGG19 to fine tune the last (5th) block of VGG19 as well as our top model(Figure 10). By freezing the first 4 CNN blocks, I freeze the hard learned pre-trained weights applicable to general image classification task. Fine tuning the most abstract layers, the 5th CNN block, so that it learns the high level features specific to this task. 

``` python
model = Sequential()

# Add VGG19 (without top layers)
for l in pretrained_model.layers:
    model.add(l)
    
# Add my Top Model 
for l in top_model.layers:
    model.add(l)

## Lock layers until the last ConvNet block
lock_until = 17  
for n, layer in enumerate(model.layers):
    if n < lock_until:
        layer.trainable = False
    else:
        layer.trainable = Tru
```
_Code Snippet 3: Freeze layers_

## __IV. Result__

## Model Evaluation and Validation

#### Benchmark Model
It took 259.6 seconds, 29 epochs to finish training the benchmark model. 8.95 seconds per epoch. On test dataset, it achieves 57% accuracy, 0.5479 kappa score and 1.83 cross-entropy loss at the 23th epoch.

![Benchmark Model Result]({{ site.url }}/assets/mlnano_capstone/model_result_benchmark1.png)
![Benchmark Model Result]({{ site.url }}/assets/mlnano_capstone/model_result_benchmark2.png)
_Figure 12: Benchmark Model Result_

#### Top Model
It took 40.83 seconds, 235 epochs to finish training the top model. Merely 0.17 second per epoch. Achieve 60% test accuracy, 0.5777 kappa score and 1.41 log loss at the 224th epoch.

![Top Model Result]({{ site.url }}/assets/mlnano_capstone/model_result_top1.png)
![Top Model Result]({{ site.url }}/assets/mlnano_capstone/model_result_top2.png)
_Figure 13: Top Model Result_

#### Fine-Tuned Model
It took 375.82 seconds, 30 epochs to finish fine tuning the model. 12.5 seconds per epoch. Achieve 80% test accuracy, 0.7890 kappa score and 0.93 log loss at the 24th epoch. Another +20% improvement in test accuracy. It's impossible to train a model from stretch to achieve 80% test accuracy with so little date. Transfer learning making it all possible with the expense of more computation time (varies depends on the benchmark and transfer model complexity).

![Combined Model Result]({{ site.url }}/assets/mlnano_capstone/model_result_combined1.png)
![Combined Model Result]({{ site.url }}/assets/mlnano_capstone/model_result_combined2.png)
_Figure 14: Fine-Tuned Model Result_

## Justification

As shown in Table 1, the power of transfer learning is already showing in the top model. Training the top model is 53x faster while achieving a better result. 
The fine-tune model significantly improve the model in all metrics.

|Metrics on Test Dataset|Training Time(sec)/Epoch|Cross-Entropy Lost|Accuracy|Kappa Score|
|Benchmark Model|8.95|1.83|57%|0.5479|
|Top Model|0.17|1.41|60%|0.5777|
|Find-Tuned Model|12.5|0.93|80%|0.7890|

_Table 1 Model Evaluation_


## __V. Conclusion__

## Free-Form Visualization

I use Global Average Pooling(GAP) instead of Fully Connected(FC) layers not only to combat overfitting also because it brings an interesting byproduct - weights of the last CNN can be used for nearly effortless object localization. [7][8]

The idea behind is simple. When using GAP layer before the final FC layer. The weights (512x1 vector - blue dotted lines below) connecting the GAP to each of the output node, e.g. panda, tell us which of those 512 feature maps activates the output the most. And each of the 512 nodes in GAP layer is calculated averaging a 7x7 feature map, green dotted lines below.

![Object Localization Explained]({{ site.url }}/assets/mlnano_capstone/object_localization_explained.png)
_Figure 15: Zoom into the last CNN - GAP - FC layers_

Therefore multiplying 7x7x512 by 512x1 gives us the weighted feature map - the activation map which allow us to visualize which part of the image is activating that particular output class. 

I randomly picked 15 classes and download pictures for each class. Figure 16 shows that the model correctly classified all new images and also showing what the CNN is looking at when predicting the class label. For example, The elephant's body and nose; The patten on the body of zebra and leopard; The black and white of panda, etc. These are strong evidences the model is learning to identify animals by identifying them by their key difference from the others.

![What CNN is looking at]({{ site.url }}/assets/mlnano_capstone/what_cnn_is_looking_at.png)
_Figure 16: CNN object localization_


## Reflection
This study demonstrated how powerful transfer learning is by outperforming a non-trivial CNN model trained from stretch with just a fraction of the training time yet achieve a better result when the training data is insufficient. And how pre-trained model can be further fine-tuned to obtain massive performance improvement that could not otherwise be achievable training a new CNN model when longer training time is acceptable. 

I can see huge potentials applying transfer learning in real life applications especially in testing out new ideas quickly without having to collect a huge amount of train data and a long training time.

I have also demonstrated an interesting byproduct of CNN model when using GAP layer between the last Convolutional layer and the FC output layer. The activation map serves as both object localization as well as a mean of model visualization which shows me how the model makes prediction. 

## Improvement
The final model can be further improved by unfreezing more Convolutional layers in the fine-tune process to further tweak the abstract layers towards our goal. It will produce better result than our final model however it will take substantially longer to train. And there will be a point where unfreezing more layers will not give us better result due to the insufficient of training data. 

When dealing with images with multiple objects, we could use the top N highest scored (not just the best) FC weights to build the activation map to obtain the location of objects recognized by the model. Then further feed the sub-images containing the objects through the CNN again to obtain the classes for all bounding boxes. Other studies have been made to utilize this method for more robust object localization, such as the study by Singh and Lee [9] where they expand the bounding box by randomly turning off parts of the training images to force the network to learn other parts of the object (not just the most iconic part like the panda face in our example).

## Reference
[1] http://yann.lecun.com/exdb/publis/pdf/lecun-99.pdf <br/>
[2] https://www.cs.princeton.edu/~rajeshr/papers/icml09-ConvolutionalDeepBeliefNetworks.pdf <br/>
[3] https://www.csc.kth.se/~heydarma/Datasets.html <br/>
[4] https://arxiv.org/pdf/1311.2901.pdf <br/>
[5] https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html <br/>
[6] https://www.kaggle.com/wiki/LogLoss <br/>
[7] http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf <br/>
[8] https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/ <br/>
[9] http://krsingh.cs.ucdavis.edu/krishna_files/papers/hide_and_seek/my_files/iccv2017.pdf
[10] https://arxiv.org/pdf/1207.0580.pdf
[11] R Hahnloser, R. Sarpeshkar, M A Mahowald, R. J. Douglas, H.S. Seung (2000). Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. Nature. 405. pp. 947–951.
[12] http://www.deeplearningbook.org/contents/convnets.html