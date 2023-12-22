# ResNet50-Based Face Recognition System

## Overview

This system use ResNet50, a deep learning model, for facial recognition tasks. It is specifically designed to generate embeddings of facial images, which can be used for various applications like identity verification. The system is trained using the Labeled Faces in the Wild (LFW) dataset, a large-scale database of face images. A unique feature of this system is its web interface, allowing users to interact with the model via their camera.

## Dataset Preprocessing

The dataset preprocessing step involves resizing and saving images from the LFW dataset. The system counts the number of face images in each subfolder, focusing on those with two or more photos. We uses a pretrained opencv face detection model to crop faces from images and then resizes them to a uniform size. This preprocessing step is crucial for maintaining consistency in the input data, ensuring the model trains effectively.

## Model Architecture

The core of this system is built on the ResNet50 model, adapted for the face recognition task. The model:

- **Loads and preprocesses** images to fit the required input size.
- **Generates triplets** (anchor, positive, and negative) for training. A triplet consists of an anchor image, a positive image (same person as the anchor), and a negative image (different person).
- **Embeds images** using a modified ResNet50 architecture. The network uses global average pooling and a dense layer to produce embeddings.
- **Employs triplet loss** for training, which helps in learning discriminative features for each identity.

## Training Process

The model is trained on the processed LFW dataset using the triplet loss function. This method focuses on minimizing the distance between an anchor and a positive (same person) while maximizing the distance between the anchor and a negative (different person). The training process is adjustable in terms of batch size, steps per epoch, and total epochs to accommodate different dataset sizes and training requirements.

```
Model is compiled and ready to be trained.
Starting training...
Epoch 1/10
100/100 [==============================] - 843s 8s/step - loss: 1.9330
Epoch 2/10
100/100 [==============================] - 843s 8s/step - loss: 1.9330
Epoch 2/10
100/100 [==============================] - 795s 8s/step - loss: 0.4049
Epoch 3/10
100/100 [==============================] - 795s 8s/step - loss: 0.4049
Epoch 3/10
100/100 [==============================] - 738s 7s/step - loss: 0.3983
Epoch 4/10
100/100 [==============================] - 742s 7s/step - loss: 0.3562
Epoch 5/10
100/100 [==============================] - 744s 7s/step - loss: 0.3197
Epoch 6/10
100/100 [==============================] - 703s 7s/step - loss: 0.2769
Epoch 7/10
100/100 [==============================] - 714s 7s/step - loss: 0.2142
Epoch 8/10
100/100 [==============================] - 554s 6s/step - loss: 0.1768
Epoch 9/10
100/100 [==============================] - 600s 6s/step - loss: 0.1704
Epoch 10/10
100/100 [==============================] - 565s 6s/step - loss: 0.1478
Training completed.
Model saved successfully.
```

## Web Interface

A key component of this system is its web interface, which enables users to use their camera for real-time face recognition. This feature enhances the applicability of the model in practical scenarios, allowing for seamless integration into existing systems or for creating new user-focused applications.

## Prediction
Prediction between 2 pictures of differents people.
```
Model loaded successfully.
1/1 [==============================] - 3s 3s/step
Embedding for the image:
[[0.55527127 0.         0.         0.         0.48789352 0.
  ...                   ... 
  0.         0.2939614  0.         0.         0.         0.        ]]
1/1 [==============================] - 0s 65ms/step
Embedding for the image:
[[0.55527127 0.         0.         0.         0.48789352 0.
  ...                   ... 
  0.         0.2939614  0.         0.         0.         0.        ]]
Euclidean Distance: 2.1662397
```

Prediction between 2 pictures of the same people.
```
Model loaded successfully.
1/1 [==============================] - 3s 3s/step
Embedding for the image:
[[0.55527127 0.         0.         0.         0.48789352 0.
    ...         ...
  0.         0.2939614  0.         0.         0.         0.        ]]
1/1 [==============================] - 0s 57ms/step
Embedding for the image:
[[0.55527127 0.         0.         0.         0.48789352 0.
  ...           ...
  0.         0.2939614  0.         0.         0.         0.        ]]
Euclidean Distance: 0.36959004
```

## Usage

Download the requirement.txt packages and launch app.py
