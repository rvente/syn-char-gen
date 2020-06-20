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

from tqdm import tqdm

IM_W = IM_H = 32

def vectorize(small_img: np.ndarray):
  """flatten method for linear classifiers
  """
  x_dims,y_dims = (small_img.shape)
  vector = np.reshape(small_img, (1,x_dims*y_dims) )
  return vector

def preproc(img: str):
  """ read in and resize """
  full_img = cv2.imread(img)/255.0
  small_img = cv2.resize(full_img, dsize=(IM_W,IM_H), interpolation=cv2.INTER_AREA)
  return (small_img)

def makeNeuralModel(num_labeled_examples):
  import tensorflow as tf
  from tensorflow.keras import datasets, layers, models
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(num_labeled_examples))
  model.summary()

  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  return model

def plot_sample_predictions(model, train_images, predict_dictionary, test_images = None):
  """TODO: refactor model prediction logic so that it can work generally
  by just taking in a train images vector, and a proposed labels vector"""

  if test_images is None:
    test_images = np.array(train_images)
  np.random.shuffle(test_images)
  predictions = np.argmax(model.predict(test_images[:25,:,:]), axis=1)
  plt.figure(figsize=(10,10))

  for i in range(25):
      plt.subplot(5,5,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(test_images[i,:,:], cmap=plt.cm.binary)
      plt.xlabel(predict_dictionary[predictions[i]])
  plt.show()

def load_database():
  imgs = sorted(glob("./png_normalized/*.png"))
  labels = ([img.split("/")[-1][0] for img in imgs])
  num_labeled_examples = len(imgs)
  ic(f"There are {num_labeled_examples} examples.")
  train_images =  np.reshape(np.array([ preproc(img) for img in imgs ]) , (num_labeled_examples, IM_H, IM_W, 3) )
  predict_dictionary = {idx: lbl for idx, lbl in zip(range(num_labeled_examples), labels)}
  # ic(labels)
  # ic(predict_dictionary)
  train_labels = np.reshape(np.array(list(range(52))), (num_labeled_examples, 1))
  # ic(train_images.shape)
  # ic(train_labels.shape)
  # ic(train_images[:5])
  return (num_labeled_examples, train_labels,
          train_images, predict_dictionary) 

def knnModel(train_images, train_labels, test_images):
  trImages = (train_images)
  tImages = test_images
  trLabels = train_labels
  tLabels = trLabels
  paramk = 7 # parameter k of k-nearest neighbors
  numTrainImages = np.shape(trLabels)[0] # so many train images
  numTestImages = np.shape(test_images)[0] # so many test images

  arrayKNNLabels = np.array([])
  for iTeI in range(0,numTestImages):
    arrayL2Norm = np.array([]) # store distance of a test image from all train images
    for jTrI in range(numTrainImages):  
      l2norm = np.sum(((trImages[jTrI]-tImages[iTeI]))**2)**(0.5) # distance between two images; 255 is max. pixel value ==> normalization   
      arrayL2Norm = np.append(arrayL2Norm, l2norm)

    sIndex = np.argsort(arrayL2Norm) # sorting distance and returning indices that achieves sort
    kLabels = trLabels[sIndex[0:paramk]] # choose first k labels  
    (values, counts) = np.unique(kLabels, return_counts=True) # find unique labels and their counts
    arrayKNNLabels = np.append(arrayKNNLabels, values[np.argmax(counts)])

  return list(map(int,arrayKNNLabels))

    # print(arrayL2Norm[sIndex[0]], kLabels, arrayKNNLabels[-1], tLabels[iTeI])
    
    # if arrayKNNLabels[-1] != tLabels[iTeI]:

    #   plt.figure(2)
    #   plt.imshow(tImages[iTeI])
    #   plt.draw()
      
    #   for i in range(numTrainImages):
    #     if trLabels[i] == arrayKNNLabels[-1]:
    #       plt.figure(1)
    #       plt.imshow(trImages[i])
    #       plt.draw()
    #       break
    
    # plt.show()



def runSVM(train_images, train_labels, test_images = None, test_labels = None):
  #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
  from sklearn.svm import SVC
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler

  # initialize svm
  clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

  num_train = train_images.shape[0]
  num_test = test_images.shape[0]
  ic(num_train, num_test)
  # flatten vectors for traning and inference
  svm_train_images = np.reshape(train_images, (num_train, IM_H*IM_W*3) )
  svm_train_labels = np.reshape(train_labels, (num_train, ))
  clf.fit(svm_train_images, svm_train_labels)

  svm_test_images = np.reshape(test_images, (num_test, IM_H*IM_W*3) )
  return clf.predict(svm_test_images)

 
def rand_augment(image):
  
  #https://stackoverflow.com/questions/41174769/additive-gaussian-noise-in-tensorflow
  def gaussian_noise_layer(input_layer, std=.05):
    noise = tf.random.normal(shape=tf.shape(input_layer), mean=.5, stddev=std, dtype=tf.float64) 
    return input_layer + noise
  imdims = image.shape
  image = 1-tf.image.resize_with_crop_or_pad(1-image, 38, 38) # Add 6 pixels of padding
  image = tf.image.random_crop(image, size=imdims) # Random crop back to 28x28
  image = gaussian_noise_layer(image)
  image = tf.image.random_brightness(image, max_delta=0.2) # Random brightness
  image = tf.clip_by_value(image, 0.0, 1.0) # clip to range
  # image = tf.contrib.image.rotate(image, tf.random_uniform(shape=[batch_size], minval=-0.3, maxval=0.3, seed=mseed), interpolation='BILINEAR')
  image = tf.image.random_contrast(image, 0.7, 1.7)
  image = zoom(image)
  # plt.figure()
  # plt.imshow(image)
  # plt.show()
  return image

# https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/
def zoom(x):
    import tensorflow as tf

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(32, 32))
        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))

def gen_augmented_dataset(train_images, train_labels, n_new_examples=5):

  augmented_data = train_images
  augmented_labels = train_labels

  for img, label in tqdm(zip(train_images, train_labels)):
    for _ in range(n_new_examples):
      augmented_labels = np.append(augmented_labels, label)
      augmented_data = np.append(augmented_data, np.array([rand_augment(img)]), axis=0)

  np.reshape(augmented_data, (-1,32,32,3))
  ic(augmented_data.shape)
  return augmented_data, augmented_labels

if __name__ == "__main__":
  (num_labeled_examples, train_labels,
      train_images, predict_dictionary) = load_database()

  trainNeuralModel = True
  saveModel = trainNeuralModel
  loadNeuralModel = False
  runknnModel = False
  runsvm = False

  # # illustrative data augmentation
  # gen_augmented_dataset(train_images, train_labels, n_new_examples=10)
  try:
    print("Attempting to load augmented data")
    agmn_train = np.load("augmented_images_train.npy")
    augmn_label = np.load("augmented_images_train_labels.npy")
    print(f"Train Images found and loaded {agmn_train.shape}")
  except:
    print("Generating examples as none were found...")
    agmn_train, augmn_label = gen_augmented_dataset(train_images, train_labels, n_new_examples=40)
    np.save("augmented_images_train.npy", agmn_train)
    np.save("augmented_images_train_labels.npy", augmn_label)
    print(f"Generated array of dims {agmn_train.shape}")

  try:
    print("Attempting to load test set")
    test_images = np.load("augmented_images_test.npy")
    test_labels = np.load("augmented_images_test_labels.npy")
    print(f"Test set found and loaded {test_images.shape}")
  except:
    print("Generating test examples as none were found...")
    test_images, test_labels = gen_augmented_dataset(train_images, train_labels, n_new_examples=10)
    np.save("augmented_images_test.npy", test_images)
    np.save("augmented_images_test_labels.npy", test_labels)
    print("Test Images were generated")

  if trainNeuralModel:
    print("Begining training")
    print(f"Training on data with dims {train_images.shape}")
    print(f"Testing on data with dims {test_images.shape}")

    model = makeNeuralModel(num_labeled_examples)
    history = model.fit(agmn_train, augmn_label, epochs=50 ,
      validation_data=(test_images, test_labels))

    plot_sample_predictions(model, test_images, predict_dictionary)

    if saveModel:
      tf.keras.models.save_model(
        model, "./nnModel", overwrite=True, include_optimizer=True, save_format=None,
        signatures=None, options=None
        )

  if loadNeuralModel:
    model = tf.keras.models.load_model("./nnModel/")
    model.summary()
    plot_sample_predictions(model, train_images, predict_dictionary, test_images=test_images)

  if runknnModel:
    predictions = knnModel(agmn_train, augmn_label, test_images)
    print(test_labels)
    print(predictions)
    print(f"Accuracy is {np.sum(test_labels == predictions)/len(predictions)}")


  if runsvm:
    predictions = runSVM(agmn_train, augmn_label, test_images, test_labels)
    print(f"Accuracy of SVM Classifier is {np.sum(test_labels == predictions)/len(predictions)}")
    print(predictions, predictions.shape)


