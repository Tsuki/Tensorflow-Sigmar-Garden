"""Script to illustrate inference of a trained tf.estimator.Estimator.

NOTE: This is dependent on mnist_estimator.py which defines the model.

mnist_estimator.py can be found at:

https://gist.github.com/peterroelants/9956ec93a07ca4e9ba5bc415b014bcca

"""

import numpy as np

import tensorflow as tf

from testing2 import get_estimator

# Set default flags for the output directories

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(

    flag_name='saved_model_dir', default_value='./mnist_training',

    docstring='Output directory for model and training stats.')

params = tf.contrib.training.HParams()  # Empty hyperparameters

# Set the run_config where to load the model from

run_config = tf.contrib.learn.RunConfig()

run_config = run_config.replace(model_dir=FLAGS.saved_model_dir)

# Initialize the estimator and run the prediction

estimator = get_estimator(run_config, params)


def predict(image):
    return estimator.predict(input_fn=image)
