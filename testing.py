import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Parameters
from tensorflow.contrib.learn import SKCompat

import main

tf.logging.set_verbosity(tf.logging.DEBUG)
learning_rate = 0.01
num_steps = 20
batch_size = 128
display_step = 100
model_path = "model.ckpt"
# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
num_input = 33 * 33  # MNIST data input (img shape: 28*28)
num_classes = 16  # MNIST total classes (0-9 digits)
image, label = main.sample()


def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    print(x_dict)
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

    export_outputs = {'predict_output': tf.estimator.export.PredictOutput(
        {"pred_output_classes": pred_classes, 'probabilities': tf.nn.softmax(logits)})}
    pred_probas = tf.nn.softmax(logits)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes,
                                          export_outputs=export_outputs)

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
        eval_metric_ops={'accuracy': acc_op},
        export_outputs=export_outputs
    )

    return estim_specs


input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': image}, y=label,
    batch_size=batch_size, num_epochs=None, shuffle=True)

# Build the Estimator
model = tf.estimator.Estimator(model_fn)
model.train(input_fn, steps=num_steps)
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': image}, y=label,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
model.evaluate(input_fn)
print(input_fn())


# Save
def serving_input_receiver_fn():
    serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None], name='input_tensors')
    receiver_tensors = {"predictor_inputs": serialized_tf_example}
    feature_spec = {"images": tf.FixedLenFeature([25], tf.float32)}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


model.export_savedmodel("export", serving_input_receiver_fn)
# Predict single images
n_images = 4
# Get images from test set
test_images = image[:n_images]
test_label = label[:n_images]
# Prepare the input data
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': test_images}, shuffle=False)
# Use the model to predict the images class
preds = list(model.predict(input_fn))

# Display
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [33, 33]), cmap='gray')
    plt.show()
    print("Model prediction:", preds[i])
    print("Model Answer:", test_label[i])
