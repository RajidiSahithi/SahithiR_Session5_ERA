# RECOGNIZATION OF HAND WRITTEN DIGITS USING CNN
Classifing hand-written digits using Convolution Neural Networks(CNN)

## Description

One of the common applications od Computer Vision and Deep Lerning is to recognize hand-written didgita from the images. The MNIST data set is a popular benchmark for this task.

- The motivation behind doing this project is to improve the accuracy of handwritten digit recognition by using CNN.
- Reason for for choosing CNNS is it achieved remarkable results in application field due to significant acheivement acquired computer technology.However, handwritten recognition still has great development space due to its complexity.
- Using CNNs and scheduler, we tried to improve the accuracy.
- I learned that CNNs is very effective in perceiving the structure of handwritten numbers in the way that help in automatic extraction of distinct fetaures and amke CNN the most suitable approach for solving handwritten recognization problems.

## STEPS INVOLVED 

- [PREPARE_MODEL](#prepare_model)
- [BUID_MODEL](#build_model)
- [TRAIN_MODEL](#train_model)
- [ANALYSE_MODEL](#analyse_model)

## PREPARE_MODEL
The model is prepared by importing datasets and transforms from torchvision. The availability of the GPU is checked.
### DATA SET
The MNIST (Modified National Institute of Standards and Technology) dataset is used.It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

A set of transformations is applied to the training data before it is fed into the neural network model . The transformations include random center cropping, resizing to 28x28 pixels with a probability of 10%, random rotation between -15 and 15 degrees, conversion to tensor format and normalization .

A set of transformations that will be applied to the test data before it is fed into the neural network model . The transformations include conversion to tensor format and normalization .

The train_data dataset is created by setting train=True, while test_data is created by setting train=False. Both datasets are downloaded from the specified directory and transformed using their respective transformations.

The batch_size variable is set to 512. 

The test_loader and train_loader are created using the DataLoader class from PyTorch. Both data loaders are passed their respective datasets and the kwargs dictionary.The kwargs dictionary contains additional parameters for the data loaders such as shuffling the data and using multiple workers for loading the data.

In this case length of train_loader is 118 and test_loader is 20.

This is the screenshot of sample train_loader

![alt text](https://github.com/RajidiSahithi/SahithiR_Session5_ERA/blob/main/images/train_loader_sample.png)
   

## BUILD_MODEL

### BUILDING A CONVOLUTIONAL NETWORK

A calss named Net is created N that inherits from the nn.Module class. The class Net is initialized and super() function returns a temporary object of the superclass, which allows you to call its methods. 

Convolutional Network is built using Conv2d(1, 32, kernel_size=3) i.e.,maxpooling2d and linear (Fully Connected) Layers.
<br/>
**CONVOLUTION LAYER:** This layer extracts high-level input features from input data and passes those features to the next layer in the form of feature maps.
<br/>
**POOLING LAYER:** It is used to reduce the dimensions of data by applying pooling on the feature map to generate new feature maps with reduced dimensions. PL takes either maximum or average in the old feature map within a given stride.
<br/>
**FULLY CONNECTED LAYER:** Finally, the task of classification is done by the FC layer. Probability scores are calculated for each class label by a popular activation function called the softmax function.

### FORWARD PASS OF NEURAL NETWORK MODULE
input X is sent to conv1 and then sent to relu (non-linearity function),then X is sent to conv2 and maxpool2d then sent to relu,then X is sent to conv3  then sent to relu,
then X is sent to conv4 and maxpool2d then sent to relu, then X is sent to Fully Connected layer then sent to relu, and again x is sent to fully connected layer.

The output of the last convolutional layer is flattened and passed through a fully connected layer with a log softmax activation function which is commonly used in image classification tasks
The class Net()  is instantiated and The object created by this instantiation is then moved to the device specified by the device variable. 
The following is the summary of the device.

<pre>
---------------------------------------------------------------------------------
 **          Layer(Type)           Output Shape            Param #           **   
=================================================================================
 **           Conv2d-1            [-1,32,26,26]                 320          **
 **           Conv2d-1            [-1,64,24,24]              18,496          **
 **           Conv2d-3           [-1,128,10,10]              73,856          **
 **           Conv2d-4             [-1,256,8,8]             295,168          **
 **           Linear-5                  [-1,50]             204,850          **
 **          Linear-6                  [-1,10]                 510           **
 ===============================================================================
 **          Total params: 593,200
 **          Trainable params :593,200
 **           Non-trainable params: 0
 --------------------------------------------------------------------------------
 **           Input size (MB) :0.00
 **           Forward/backward pass size (MB):0.67
 **           Params size (MB): 2.26
 **           Estimated Total Size (MB) :2.94
 ----------------------------------------------------------------------------------
</pre>

## TRAIN_MODEL

The tqdm library is used to create a progress bar for the training loop.The train() function is a custom function in Python that takes as input a model, device, train_loader, optimizer and epoch. It trains the model by iterating over the train_loader and updating the model parameters using backpropagation

<br/>
**Loss:**  can be used to estimate the loss of the model so that the weights can be updated to reduce the loss on the next evaluation.We use fit method then training starts.
<br/>
**Optimizer:** controls the learning rate. We will be using ‘SGD’ as our optmizer with learning rate 0.01 and and momentum 0.9
<br/>
**scheduler:** is used to adjust the learning rate. This is done for 20 epochs

## ANALYSE_MODEL
The model is analysed based on Train_Loss,Train_Accuracy and Test_loss and Test_Accuracy


![alt text](https://github.com/RajidiSahithi/SahithiR_Session5_ERA/blob/main/images/loss_acc.png)

## Features

Below is the table for Receptive Field_IN, Number of Input Channels, J_IN, Stride Size, Kernal Size,Receptive Field_OUT, Number of Output Channels 
Padding=0
| Layer   | RF_IN | N-IN   | J_IN  |  s  |  k  | RF_OUT | N_OUT |
|--------:|-------|--------|-------|-----|-----|--------|-------|
| Conv1   |   1   |   28   |   1   |  1  |  3  |    3   |   26  |
| Conv2   |   3   |   26   |   1   |  1  |  3  |    5   |   24  |
| Maxpool |   5   |   24   |   1   |  2  |  2  |    6   |   12  |
| Conv3   |   6   |   12   |   2   |  1  |  3  |   10   |   10  |
| Conv4   |   10  |   10   |   2   |  1  |  3  |   14   |    8  |
| Maxpool |   14  |    8   |   2   |  2  |  2  |   16   |    4  |

## Tests

This project can be tested by changing the batch size, kernal size of convolutional Network, learning rate, momentum. 
