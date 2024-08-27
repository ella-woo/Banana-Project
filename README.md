# Banana Project
## Overview
This is a Banana Ripeness Detector. The model looks at a picture of a banana and classifies it into one of three categories: Ripe, Unripe, Overripe. It is an image classification network that uses the resnet18 CNN(Convolutional Neural-Network). 
## How it works
## Running this project
1. clone this repository under the jetson-inference folder in your nano. 
2. download the resnet18.onnx file model from this link (inset google drive link) and put it in the repository. 
3. in the terminal, navigate to this folder where banana.py is located and run this command and then replace overripe.jpg with the banana picture that you want to classify.

python3 banana.py --network=/home/nvidia/jetson-inference/Banana\ Project/resnet18.onnx --labels=/home/nvidia/jetson-inference/Banana\ Project/labels.txt /home/nvidia/jetson-inference/Banana\ Project/overripe.jpg



## Materials
- laptop 
- nano 
- internet connection 

## Video Tutorial
