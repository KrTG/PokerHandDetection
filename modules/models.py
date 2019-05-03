from modules.const import *

import tensorflow as tf

def dnn_model_fn(features, labels, mode):
    # input layer
    input_layer = features["x"]

    # convolutional layer #1
    dense = tf.layers.dense(
        inputs=input_layer,
        units=1024,
        activation=tf.nn.relu,
    )
    
    # logits layer
    logits = tf.layers.dense(inputs=dense, units=NUM_CLASSES)
    softmax = tf.nn.softmax(logits, name="softmax_tensor")

    predictions = {
        "image": input_layer,  
        "dense": dense,
        "logits": logits,                
        "probabilities": softmax,
        "classes": tf.argmax(input=logits, axis=1),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        lrate = tf.get_variable("lrate", initializer=tf.constant(LEARNING_RATE), dtype=tf.float32)      
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lrate,
            beta1=0.9,
            beta2=0.999,
            epsilon=0.1,
        )
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluations metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops
    )

def cnn_model_fn(features, labels, mode):
    """Model function for CNN"""
    # input layer
    input_layer = tf.reshape(features["x"], [-1, ISIZE, ISIZE, 1])

    # convolutional layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    # pooling layer #1
    pool1 = tf.layers.max_pooling2d(
        name='pooling1',
        inputs=conv1,
        pool_size=[2, 2],
        strides=2 
    )

    # convolutional layer #2 and pooling layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(
        name='pooling2',
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    # dense layer
    pool2_flat = tf.reshape(pool2, [-1, ISIZE//4 * ISIZE//4 * CHANNELS * 128])
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu,
    )
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.7, training=(mode == tf.estimator.ModeKeys.TRAIN)
    )

    # logits layer
    logits = tf.layers.dense(inputs=dropout, units=NUM_CLASSES)

    predictions = {
        "image": input_layer,        
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.identity(loss, name="loss")

    if mode == tf.estimator.ModeKeys.TRAIN:
        lrate = tf.get_variable("lrate", initializer=tf.constant(LEARNING_RATE), dtype=tf.float32)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lrate,
            beta1=0.9,
            beta2=0.999,
            epsilon = 0.1,
        )
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluations metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops
    )