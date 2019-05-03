from modules.const import *

import tensorflow as tf
import numpy as np
import cv2

import os
import glob
import random

def get_image(path):
    image = cv2.imread(path)    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (ISIZE, ISIZE))
    image = np.float32(image) / 255
    return image

def get_label(path):
    folder_path = os.path.split(path)[0]
    label = int(os.path.basename(folder_path))
    return label

def get_dataset_paths(dataset_path, with_labels=True):
    if with_labels:
        path_to_match = os.path.join(dataset_path, "*", "*.png")
    else:
        path_to_match = os.path.join(dataset_path, "*.png")
        
    paths = glob.glob(path_to_match)        
    paths = sorted(paths, key=lambda x: int(os.path.basename(x).split(".")[0].split("-")[0]))
    
    return paths

def generator_from_dataset(dataset_path, shuffle, with_labels=True):    
    if with_labels:
        path_to_match = os.path.join(dataset_path, "*", "*.png")
    else:
        path_to_match = os.path.join(dataset_path, "*.png")
    paths = glob.glob(path_to_match)        

    if shuffle:    
        random.shuffle(paths) 
    else:
        paths = sorted(paths, key=lambda x: int(os.path.basename(x).split(".")[0].split("-")[0]))

    def gen():
        for path in paths:
            if with_labels:
                yield ({"x": get_image(path)}, get_label(path))
            else:
                yield {"x": get_image(path)}
    return gen

def generator_from_images(grouped_images):
    def gen():
        for group in grouped_images:
            for image in group:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (ISIZE, ISIZE))
                image = np.float32(image) / 255
                yield {"x": image}    
    return gen

def labeled_input_fn(path, batch_size, num_epochs, shuffle, generator=None):
    if generator is None:
        generator = generator_from_dataset(path, shuffle=shuffle, with_labels=True)

    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_types=(({"x": tf.float32}, tf.int64)),
        output_shapes=(({"x": tf.TensorShape([ISIZE, ISIZE])}, tf.TensorShape([]))),
    )       
    if shuffle:
        dataset = dataset.shuffle(batch_size * 10)

    dataset = dataset.repeat(num_epochs)    
    dataset = dataset.batch(batch_size)       
    return dataset

def unlabeled_input_fn(path, batch_size, num_epochs, shuffle, generator=None):
    if generator is None:
        generator = generator_from_dataset(path, shuffle=shuffle, with_labels=False)
        
    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_types=({"x": tf.float32}),
        output_shapes=({"x": tf.TensorShape([ISIZE, ISIZE])}),
    )       
    if shuffle:
        dataset = dataset.shuffle(batch_size * 10)    
    dataset = dataset.repeat(num_epochs)    
    dataset = dataset.batch(batch_size)       
    return dataset
