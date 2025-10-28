# YOLO V11 Retraining - Team 103

This notebook can be used as either a template for creating future object detection training programs, or can be used on different datasets to make future models. 

Our goal here is to take the pretrained YOLO V11, or other models, which have robust object detection capabilities, and introduce our new data to the model. This introduction of new data is the same process as originally training the neural network. We will pass in labeled data, and the AI will predict what those labels should be, based on the input image. The Network will then take the "Loss" or error between predicted label, and actual label, and use that to update the network weights. This is called back-propagation.

AI like this is a network, made to mimic what people thought the human brain was 60 years ago. Now we do know that they were incredibly wrong on the neuroscience side, however on the electrical engineering side, this paradigm introduced a new type of computing with the introduction of the perceptron. 

![image.png](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgxIM5ebUtSZ06KqB3p1Q3L1s_6pMKr0riYmEoj1-xyuT3kzDBOvxj9K9WJp-koQBvHe8BDtGH-PTjp8Gf9qku1Hj3l4XJbRXPAwRhSL6kHuXgCEy_cL09ri11hCmzRBltxpka1MgJbaARsI8PpijnMwcarTCn68i47xoeMEPKH2ngutLA0XLuYk0erpA/s1640/single%20and%20multi-layer%20perceptron%20image%20combined%202.png)

The Perceptron was the first Neural Network created, and was utilized to identify hand written numbers by reporting 10 outputs. Each output was tied to a digit, such as "5," and could be a number 0 to 1, which represented the probability that "5" or any other digit, was the number on the inputed image. 
The idea behind the perceptron was to take in each individual input, represented at the input layer, and pass each one on to a hidden layer. Each hidden layer was made of some number of "neurons." Each neuron would sum all the values from the layer before, with each value multiplied by some "weight."

Each neuron has a table of weights, the length of which is identical to the layer before, so that an output per neuron can be generated as: 

Neuron_Output = X1*W1 + X2*W2 ... + Xn*Wn

Now this output could be any number, however for us, it is more stable and understandable if we limit these outputs on any neuron to 0 through 1. We use a "sigmoid" function to do this, which is a function defined so that any real number is mapped to 0 through 1.

![image.png](https://miro.medium.com/v2/resize:fit:1400/1*a04iKNbchayCAJ7-0QlesA.png)


## What does any of this do for us
### (Currently nothing)

So we have inputs, and we know that each neuron has some list of weights, and we know that there are layers of neurons all doing this on the layer before. So what. 

The main theory that holds this system together, is that we can aproximate ANY function at the output, if we have enough neurons, and can find the correct weights for those neurons. "Enough" generally means infinite here, and finding the "correct" weights for any of those neurons would also take "infinite" time... But as with everything, good enough works! Largely the definition of "enough" will change widely depending on application. "Enough" in terms of classifying whether an image shows a handwritten "6" is rougly 60 neurons, with up to 60^2 parameters (3600). "Enough" for drawing a box around a game piece is more like 3 million parameters..

As we add parameters, or weights, we require more data in and more training time to aproximate our outputs. But how does this work? Right now we just have random "weights" and no way to find them.

## Back Propagation

Backpropagation is a method used in training neural networks where the model:

1: Makes a prediction
2: Compares it to the correct answer
3: Calculates the error
4: Sends that error backward through the network
5: Adjusts the internal settings (weights) to reduce future errors

We do this process over and over during the training phase, on our input training data. We may have started with randomly initialized "weights,"  but after many training rounds, these wieghts will update and aproach the "correct" values. Correct is in quotes here, and that is on purpose. Correct to the neural network could mean a lot of things, and none of those are what you or I would define as "correct."

If we pass in bad data that has some sort of tell, or improperly labeled data, or data that has a lot of one type of object, but not another, the network may "learn" the wrong things. If you were training a network to detect a ball, you might take many images of the ball, all in the environment you expect it to come from, say on the carpet. This makes intuitive sense to us, becuase it seems more realistic to what the camera will be seeing. The problem arises from something we call Correlation between variables. If all our pictures of balls are on carpet, then the network has no other context but to think these are correlated variables. You may end up with a detector that sees a ball and says "ball," but also sees the carpet and falsely assumes "ball."

There are also cases of this in the reverse, where the model might not detect a ball becuase it is against a wooden wall, or in the air.

All this to say that AI in every form is completely restricted by quality data with quality labels.

# TLDR Lets train our model on a given dataset

We will be assuming a few things. For one this document will be originally setup to interface with YOLOV11. This model is pretrained, and we will be adjusting the weights on it with the same concept of back propagation shown above. The advantage of using a base like YOLOv11 is that it is already quite good at detection, but of course doesn't know the game pieces we are attempting to find. So we can introduce this model to our data, rather than starting fresh, and greatly accelerate our training time, as well as decrease the amount of data we need to pass in to train all 3 million parameters. Rather than training all 3M, we just need to adjust them enough to recognize our objects as well.

## Step 1: Requirements before you run this file.

Python version 3.10 or higher.

If using an nvidia gpu for acceleration.

NVIDIA CUDA V13.0 -> if you have a different version of cuda, you will need to make sure you find matching packages pythons torch and torchvision librarys. You can install the requirements.txt, then run each of these to replace pytorch with the cuda available one:
```python
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
where the 130 at the end is for V13.0

You likely already have done this, but in this directory we need to create a virual environment and gather the required librarys for this script.

Note: python -m before commands just tells the cmd to use python to find some of these commands. often you can call some of these without the "python -m" but this should work every time.
run the following commands !! IN CMD NOT POWERSHELL !!:

```python
pip install venv ## install the virtual environment library

python -m venv env ## do this in specifically cmd, not powershell, and in the folder with this file.

.\env\Scripts\activate.bat ## takes this terminal into the special virtual environment, 
                            # allowing librarys to live here for this project specifically,
                            # rather than on the whole computer. this helps with keeping versions alligned.

 python -m pip install -r requirements.txt ## installs all the requirements for this notebook using the 
                                             # predefined list.
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 ## isntall torch and torchvision with correct versioning
```

## Step 2: Opening this document in a live terminal. 

With cmd in the virtual environment and in the folder with this file, run the command :

python -m notebook

and open up the file when propted with a browser view of the directory. You should now be able to run python commands right in this document.




