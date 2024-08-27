from jetson_inference import imageNet

import jetson.inference
import jetson.utils

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
parser.add_argument("--labels", type=str)
opt = parser.parse_args()

print (opt.filename)
print (opt.network)

img = jetson.utils.loadImage(opt.filename)

net = jetson.inference.imageNet(opt.network)

net = imageNet(opt.network) 

#net = jetson.inference.imageNet(model=opt.network, labels=opt.labels)

class_idx, confidence = net.Classify(img)

class_desc = net.GetClassDesc(class_idx)

print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))

