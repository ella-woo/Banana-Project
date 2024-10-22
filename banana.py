from jetson_inference import imageNet

from jetson_inference import imageNet
from jetson_utils import loadImage

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
#parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
#parser.add_argument("--labels", type=str)
opt = parser.parse_args()

print (opt.filename)
#print (opt.network)

img = loadImage(opt.filename)

#net = jetson_inference.imageNet(opt.network)

net = imageNet(model="resnet18.onnx", labels="labels.txt", 
               input_blob="input_0", output_blob="output_0") 

#net = jetson.inference.imageNet(model=opt.network, labels=opt.labels)

class_idx, confidence = net.Classify(img)

class_desc = net.GetClassDesc(class_idx)

print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))

