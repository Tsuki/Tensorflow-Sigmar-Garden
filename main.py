from __future__ import print_function

import random
from enum import Enum
import os

import itertools

import sys
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from Marble import Marble
from Parameters import batch_size, learning_rate, num_steps, display_step

from State import State
from utils import img_pos, edges_at, PIXELS_TO_SCAN, FIELD_POSITIONS

n_hidden_1 = int(len(PIXELS_TO_SCAN) / 3)  # 1st layer number of neurons
n_hidden_2 = int(len(PIXELS_TO_SCAN) / 3)  # 2nd layer number of neurons
num_input = len(PIXELS_TO_SCAN)  # MNIST data input (img shape: 28*28)
num_classes = len(Marble)
MARBLE_BY_SYMBOL = dict(zip([Marble.symbol(e) for e in Marble], [e.name for e in Marble]))
TRAIN_CASES = dict.fromkeys([e.name for e in Marble], [])
image = []
label = []
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("string", [None, num_classes])


def sample():
    for i in range(1, 7):
        img = Image.open(os.path.join("sample", str(i) + ".png")).convert('LA')
        samples = list(itertools.chain.from_iterable(
            [lines.split() for lines in open(os.path.join("sample", str(i) + ".txt"), "r").readlines()]))
        for j, (pos, symbol) in enumerate(zip(FIELD_POSITIONS, samples)):
            marble = MARBLE_BY_SYMBOL[symbol]
            edge_pixels = edges_at(img, *img_pos(*pos))
            image.append(edge_pixels)
            label.append(Marble[marble].value)
    images = np.array(image).astype(np.float32)
    labels = np.array(label).astype(np.int8)
    return images, labels
    # TRAIN_CASES[marble] = TRAIN_CASES[marble] + [edge_pixels]


def train():
    marble = random.choice([e.name for e in Marble])
    edge_pixels = random.choice(TRAIN_CASES[marble])
    a = list(map(lambda x: 1.0 if x in edge_pixels else 0.0, PIXELS_TO_SCAN))
    b = list(map(lambda x: 1.0 if marble is x else 0.0, [e.name for e in Marble]))


def init_image(img):
    status = State()
    for pos in FIELD_POSITIONS:
        try_edges = edges_at(img, *img_pos(*pos))
        # result = ANN.run(list(map(lambda x: 1.0 if x in try_edges else 0.0, PIXELS_TO_SCAN)))
        # marble = sorted(list(zip(result, [e.name for e in Marble])), reverse=True)[0]
        # status.state[pos] = Marble.symbol(Marble[marble[1]])
    print(status)


def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['images']
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer


def model_fn(features, labels, mode):
    # Build the neural network
    logits = neural_net(features)

    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    # pred_probas = tf.nn.softmax(logits)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


# https://stackoverflow.com/questions/44460362/how-to-save-estimator-in-tensorflow-for-later-use
def serving_input_receiver_fn():
    serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None], name='images')
    receiver_tensors = {"predictor_inputs": serialized_tf_example}
    feature_spec = {"words": tf.FixedLenFeature([25], tf.int64)}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def init():
    if os.path.exists("network.fann"):
        print("Load Network from network.fann")
        # ANN.create_from_file("network.fann")
    else:
        print("Train Network")
        sample()
        # print(image)
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': np.array(image)}, y=np.array(label), batch_size=batch_size, num_epochs=None, shuffle=True)
        print("Estimator")
        model = tf.estimator.Estimator(model_fn)
        print(model)
        model.train(input_fn, steps=num_steps)
        predict = model.predict(image[0])
        print(list(predict))
        print("evaluate")
        # Use the Estimator 'evaluate' method
        evaluate = model.evaluate(image, label)
        print(evaluate)
        print("Saveing")
        # model.export_savedmodel(os.getcwd(), serving_input_receiver_fn)
        # n_images = 10
        # Get images from test set
        # test_images = image[:n_images]
        # Prepare the input data
        # input_fn = tf.estimator.inputs.numpy_input_fn(
        #     x={'images': np.array(test_images)}, shuffle=False)
        # Use the model to predict the images class
        # preds = list(model.predict(input_fn))

        # Display
        # for i in range(n_images):
        #     plt.imshow(np.reshape(test_images[i], [33, 33]), cmap='gray')
        #     plt.show()
        #     print("Model prediction:", Marble(preds[i]))


def main():
    # print(Marble.symbol(Marble.Fire))
    tf.logging.set_verbosity(tf.logging.DEBUG)
    print()
    init()
    init_image(Image.open(os.path.join("sample", "1.png")).convert('LA'))
    # print(pixels_to_scan())
    # print(field_positions())
    # print(img_pos(1, 1))
    pass


if __name__ == '__main__':
    main()
