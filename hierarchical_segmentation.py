from cv2 import cv2
from icecream import ic
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import sklearn

import numpy as np
import pandas as pd
import subprocess
import os

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import KMeans

from tqdm import tqdm
import pickle
# import the necessary packages
import argparse
import time

# convert -density 500 source.pdf -background white -alpha remove -alpha off sample-document.png

# https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
# construct the argument parser and parse the arguments

# import the necessary packages
import imutils

def sliding_window(image, stepx, yIter, windowSize, start=(0,0), end=(1000,1000)):
	# slide a window across the image
  xMin,yMin = start
  xMax,yMax = end
  assert yMax >= yMin and xMax >= xMin
  yStarts = map(lambda x: x-windowSize[1]//2, yIter)
  for y in yStarts:
    for x in range(xMin, xMax, stepx):
      # yield the current window
      yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

if __name__ == "__main__":
  dictionary = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
  7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
  15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
  22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'a', 27: 'b', 28: 'c',
  29: 'd', 30: 'e', 31: 'f', 32: 'g', 33: 'h', 34: 'i', 35: 'j',
  36: 'k', 37: 'l', 38: 'm', 39: 'n', 40: 'o', 41: 'p', 42: 'q',
  43: 'r', 44: 's', 45: 't', 46: 'u', 47: 'v', 48: 'w', 49: 'x',
  50: 'y', 51: 'z'}
  try:
    # print("Attempting to load augmented data")
    # agmn_train = np.load("augmented_images_train.npy")
    # augmn_label = np.load("augmented_images_train_labels.npy")
    # print(f"Train Images found and loaded{agmn_train.shape}")
    # with open("dictionary.pkl", "rb") as dictfile:
    #   dictionary = pickle.load(dictfile)
    model = tf.keras.models.load_model("./nnModel/")
    model.summary()
  except Exception as e:
    print(e)
    raise RuntimeError("Please save data and model before running.")

  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--image", required=True, help="Path to the image")
  args = vars(ap.parse_args())
  # load the image and define the window width and height
  image = cv2.imread(args["image"])
  (winW, winH) = (38, 38)


  edges = cv2.Canny(image,50,150,apertureSize = 7)
  # cv2.imshow('edges', edges) # show canny edges detected 
  lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=10,maxLineGap=200)
  xList, yList = [],[]
  for line in lines:
      x1,y1,x2,y2 = line[0]
      xList.extend([x1,x2])
      yList.append(int((y1 + y2 )/2))
      # cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)

  chunk = lambda x: x//10 * 10
  skipList = map(chunk, yList)

  yArray = np.reshape(np.array(yList), (-1,1))
  xArray = np.reshape(np.array(xList), (-1,1))
  # ic(yArray, yArray.shape)
  k_means_y = KMeans(n_clusters=20).fit(yArray)
  # k_means_x = KMeans(n_clusters=2).fit(xArray)
  # find cluster centroids
  centroids = k_means_y.cluster_centers_
  # ic(centroids)
  # x0,x1 = map(int,list(k_means_x.cluster_centers_))
  x0 , x1 = 0,3000
  for textline in centroids:
    y0 = int(textline)
    # cv2.line(image, (x0,y0), (x1, y0), (0,255,0), 2)

  # xMin,xMax = map(int,(sorted((x0, x1))))
  xMin = min(xList)
  xMax = max(xList)
  yMin,yMax = map(int,(min(centroids),max(centroids)))

  stepx = 51
  yIter = sorted( map(int, centroids))
  # loop over the sliding window for each layer of the pyramid
  predictions = []
  for (x, y, window) in sliding_window(image, stepx, yIter, windowSize=(winW, winH), start=(xMin, yMin), end=(xMax, yMax)):
    # if the window does not meet our desired window size, ignore it
    if window.shape[0] != winH or window.shape[1] != winW:
      continue
    full_img = window/255.0
    small_img = cv2.resize(full_img, dsize=(32,32), interpolation=cv2.INTER_AREA).reshape(-1,32,32,3)

    a = np.argmax(model.predict(small_img), axis=1)
    pred = dictionary[int(a)]
    ic(pred)

    # since we do not have a classifier, we'll just draw the window
    clone = image.copy()
    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    cv2.imshow("Window", clone)
    cv2.waitKey(1)
    time.sleep(0.2)