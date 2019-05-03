from modules.const import *
from modules.utility import show
from modules import models
from modules import hooks
from modules import nn_input
from modules import interpret_labels

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import cv2

from collections import Counter
import glob
import os
import json
import logging
import shutil
import json

MODEL_FN = models.cnn_model_fn
INPUT_FN = nn_input.labeled_input_fn
UNLABELED_INPUT_FN = nn_input.unlabeled_input_fn

tf.logging.set_verbosity(tf.logging.INFO)
np.set_printoptions(
    precision=2,        
    suppress=True,
    floatmode="fixed",
)

def get_config():
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig(
        session_config=session_config
    )

    return run_config

def find_best_learning_rate(data_folder, num_steps):
    classifier = tf.estimator.Estimator(
        model_fn=MODEL_FN, model_dir=None, config=get_config()
    )

    lrate_hook = hooks.LearningRateFinderHook("loss", num_steps=num_steps)
    classifier.train(
        input_fn=lambda: INPUT_FN(os.path.join(data_folder, "train"), BATCH_SIZE, num_epochs=None, shuffle=True),
        hooks=[lrate_hook]
    )

    measurements = lrate_hook.get_measurements()
    measurements = np.array(measurements)
    measurements = measurements.transpose()        
    plt.plot(measurements[0], measurements[1])
    plt.xscale('log')
    plt.xlabel('learning rate')
    plt.ylabel('loss')
    plt.savefig("lrtest.png")
    plt.show()
 
def train_and_eval(data_folder, num_steps):
    biai_classifier = tf.estimator.Estimator(
        model_fn=MODEL_FN, model_dir=SAVE_PATH, config=get_config()
    )

    early_stopping_hook = hooks.EarlyStoppingHook(
        monitor='sparse_softmax_cross_entropy_loss/value',
        patience=10000,
        average=10
    )
    learning_rate_scheduler = hooks.LearningSchedulerHook(BATCH_SIZE, num_steps, LEARNING_RATE * 0.1, LEARNING_RATE, crush_rate=CRUSH_RATE)

    biai_classifier.train(
        input_fn=lambda: INPUT_FN(os.path.join(data_folder, "train"), BATCH_SIZE, num_epochs=None, shuffle=True),
        steps=num_steps,
        hooks=[early_stopping_hook, learning_rate_scheduler]
    )

    evaluate(data_folder)

def evaluate(data_folder):    
    biai_classifier = tf.estimator.Estimator(
        model_fn=MODEL_FN, model_dir=SAVE_PATH, config=get_config()
    )

    eval_results = biai_classifier.evaluate(
        input_fn=lambda: INPUT_FN(os.path.join(data_folder, "test"), 1, num_epochs=1, shuffle=False),
    )        
    train_set_results = biai_classifier.evaluate(
        input_fn=lambda: INPUT_FN(os.path.join(data_folder, "train"), 1, num_epochs=1, shuffle=False),
    )
    
    print("TRAIN SET RESULTS:")
    print(train_set_results)
    print("EVAL SET RESULTS:")
    print(eval_results)

def show_wrong_predictions(data_folder, subset):
    classifier = tf.estimator.Estimator(
        model_fn=MODEL_FN, model_dir=SAVE_PATH, config=get_config()
    )

    predictions = classifier.predict(
        input_fn=lambda: INPUT_FN(os.path.join(data_folder, subset), 1, num_epochs=1, shuffle=False),
    )
    

    paths = nn_input.get_dataset_paths(os.path.join(data_folder, subset))
    predictions = list(predictions)
    incorrect = []
    for path, prediction in zip(paths, predictions):
        true_label = nn_input.get_label(path)
        if true_label  != prediction["classes"]:
            prediction["true_label"] = true_label
            incorrect.append(prediction)
    print("{} incorrect of {}".format(len(incorrect), len(predictions)))    
            
    print("Wrong predictions:\n")
    for example in incorrect:
        true_desc = interpret_labels.get_description(example["true_label"])
        prediction_desc = interpret_labels.get_description(example["classes"])        
        print("True: {}, Predicted: {}".format(true_desc, prediction_desc))        
        show(example["image"], scaling=5)    

# accepts predictions grouped by num_corners(for each card)
def prediction_voting(predictions):
    predictions = list(predictions)
    decisions = []
    for i in range(0, len(predictions), NUM_CORNERS):
        group = predictions[i:i+NUM_CORNERS]
        counter = Counter()
        for prediction in group:
            counter[prediction['classes']] += 1            
        most_common = counter.most_common(2)
        winner_class = most_common[0][0]
        winner_votes = most_common[0][1]
        
        # If there is a tie
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:            
            best_prediction = max(group, key=lambda x: max(x['probabilities']))
            winner_class = best_prediction['classes']
                
        decisions.append({'classes': winner_class, 'votes': winner_votes,
         'predictions': group})

    return decisions

def predict(input_images):
    with tf.device('/cpu:0'):   
        biai_classifier = tf.estimator.Estimator(
            model_fn=MODEL_FN, model_dir=SAVE_PATH, config=get_config()
        )        

        predictions = biai_classifier.predict(            
            input_fn=lambda: UNLABELED_INPUT_FN(path=None, batch_size=BATCH_SIZE, num_epochs=1, shuffle=False, generator=nn_input.generator_from_images(input_images))
        )                 
        return prediction_voting(predictions)

def label_images(data_folder, output_file):
    biai_classifier = tf.estimator.Estimator(
        model_fn=MODEL_FN, model_dir=SAVE_PATH, config=get_config()
    )

    predictions = biai_classifier.predict(
        input_fn=lambda: UNLABELED_INPUT_FN(data_folder, 1, num_epochs=1, shuffle=False)
    )                 
        
    label_list = []
    predictions = list(predictions)
    for i, p in enumerate(predictions[::NUM_CORNERS]):
        color, rank = interpret_labels.get_classes(p["classes"])
        label_list.append([i, color, rank])
    
    with open(output_file, "w") as f:
        json.dump(label_list, f)

def clean():
    shutil.rmtree(SAVE_PATH, 
        onerror=lambda x,y,z: print("Couldn't remove model directory({})".format(SAVE_PATH))
    )