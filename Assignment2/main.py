import pickle

import numpy as np
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.ops.image_ops_impl import ResizeMethod
import time

NUM_LABELS = 47
rnd = np.random.RandomState(123)
tf.set_random_seed(123)


# Following functions are helper functions that you can feel free to change
def convert_image_data_to_float(image_raw):
    img_float = tf.expand_dims(tf.cast(image_raw, tf.float32) / 255, axis=-1)
    return img_float


def visualize_ae(i, x, features, reconstructed_image):
    '''
    This might be helpful for visualizing your autoencoder outputs
    :param i: index
    :param x: original data
    :param features: feature maps
    :param reconstructed_image: autoencoder output
    :return:
    '''
    plt.figure(0)
    plt.imshow(x[i, :, :], cmap="gray")
    plt.figure(1)
    plt.imshow(reconstructed_image[i, :, :, 0], cmap="gray")
    plt.show()
    # plt.figure(2)
    # plt.imshow(np.reshape(features[i, :, :, :], (7, -1), order="F"), cmap="gray", )


def build_cnn_model(placeholder_x, placeholder_y):
    with tf.variable_scope("cnn") as scope:
        img_float = convert_image_data_to_float(placeholder_x)

        # This is a simple fully connected network
        img_flattened = tf.reshape(img_float, [-1, np.prod(placeholder_x.shape[1:])])
        weight = tf.get_variable("fc_weight", shape=(img_flattened.shape[1], NUM_LABELS),
                                 initializer=tf.random_normal_initializer(stddev=0.01))
        logits = tf.matmul(img_flattened, weight)

        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=placeholder_y, logits=logits)

        # gradient decent algorithm
        params = [weight]
        learning_rate = 0.001
        grad = tf.gradients(loss, weight)[0]
        train_op = tf.assign_add(weight, -learning_rate * grad)

    return params, train_op


CNN_MODEL_PATH = "./CNN/cnn_model"
AE_MODEL_PATH = "./AE/ae_model"

# Major interfaces
def train_cnn(x, y, placeholder_x, placeholder_y):
    # TODO: implement CNN training.
    # Below is just a simple example, replace them with your own code
    num_iterations = 100

    # This is a simple model, write your own
    params, train_op = build_cnn_model(placeholder_x, placeholder_y)

    cnn_saver = tf.train.Saver(var_list=params)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        feed_dict = {placeholder_x: x, placeholder_y: y}
        for n_pass in range(num_iterations):
            sess.run(train_op, feed_dict=feed_dict)
            print("Epoch {} finished".format(n_pass))
        cnn_saver.save(sess=sess, save_path=CNN_MODEL_PATH)


def test_cnn(x, y, placeholder_x, placeholder_y):
    # TODO: implement CNN testing
    raise NotImplementedError
    return result_accuracy


def build_ae_model(placeholder_x, lr):
    img_float = convert_image_data_to_float(placeholder_x)
    # encoder
    conv1 = tf.layers.conv2d(inputs=img_float, filters=32, kernel_size=[3, 3], strides=(1, 1), padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], strides=(2, 2), padding='same', activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3], strides=(1, 1), padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding='same', strides=2)
    feature_map = pool2  # shape [-1, 4, 4, 64]
    # decoder
    depool2 = tf.image.resize_images(images=feature_map, size=[7, 7], method=ResizeMethod.NEAREST_NEIGHBOR)
    deconv3 = tf.layers.conv2d_transpose(inputs=depool2, filters=64, kernel_size=[3, 3], strides=1, padding='same', activation=tf.nn.relu)
    deconv2 = tf.layers.conv2d_transpose(inputs=deconv3, filters=32, kernel_size=[3, 3], strides=2, padding='same', activation=tf.nn.relu)
    depool1 = tf.image.resize_images(images=deconv2, size=[28, 28], method=ResizeMethod.NEAREST_NEIGHBOR)
    deconv1 = tf.layers.conv2d_transpose(inputs=depool1, filters=1, kernel_size=[3, 3], strides=1, padding='same', activation=tf.nn.relu)
    reconstructed_image = deconv1
    # loss = tf.losses.mean_squared_error(img_float, reconstructed_image)
    loss = tf.reduce_mean(tf.square(reconstructed_image - img_float))
    # print([x.name for x in tf.trainable_variables()])

    params = tf.trainable_variables()
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.9)
    train_op = optimizer.minimize(loss=loss)

    return params, train_op, loss, feature_map, reconstructed_image

def train_ae(x, placeholder_x):
    num_epoch = 20
    # learning_rate = [0.1, 0.01, 0.001]
    learning_rate = [0.1, 0.01, 0.001]
    batch_sizes = [64, 128, 256]

    # train validation split for Holdout validation
    train_x = x[0: int(0.8 * x.shape[0])]
    validation_x = x[int(0.8 * x.shape[0]):]

    # hold validation to find the best learning_rate

    best_model_loss = 1000000
    best_batch_size = batch_sizes[0]
    best_learning_rate = learning_rate[0]
    for batch_size in batch_sizes:
        for lr in learning_rate:
            params, train_op, loss, feature_map, reconstructed_image = build_ae_model(placeholder_x, lr)
            ae_saver = tf.train.Saver()
            start_time = time.time()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                for epoch in range(num_epoch):
                    num_batch = int(train_x.shape[0] / batch_size)
                    for batch in range(num_batch):
                        x_batch = train_x[batch * batch_size: (batch + 1) * batch_size]
                        feed_dict = {placeholder_x: x_batch}
                        _ = sess.run([train_op], feed_dict=feed_dict)
                    if epoch != 0:
                        loss_value = sess.run(loss, feed_dict=feed_dict)
                        print("Epoch {0}, Loss: {1}, Time:{2}s".format(epoch, loss_value, time.time()-start_time))
                # finish all the epoth, report the validation loss and compare it with current best
                loss_value = sess.run([loss], feed_dict={placeholder_x: validation_x})
                print("lr={0},batch_size={1}, Validation Loss: {2}, Time: {3}s".format(lr, batch_size, loss_value, time.time() - start_time))
                if loss_value[0] < best_model_loss:
                    best_model_loss = loss_value
                    best_learning_rate = lr
                    best_batch_size = batch_size
                    ae_saver.save(sess, save_path=AE_MODEL_PATH)
                    print("Update best model with lr {0}, batch size {1}".format(lr, batch_size))
    print("Model with learning_rate {0}, batch size {} got the best performance, loss value is {1}".format(
        best_learning_rate,
        best_batch_size,
        best_model_loss))
    config = {}
    config['lr'] = best_learning_rate
    config['batch_szie'] = best_batch_size
    with open('config_ae.pkl', 'wb') as handle:
        pickle.dump(config, handle)

def evaluate_ae(x, placeholder_x):
    with open('config_ae.pkl', 'rb') as f:
        config = pickle.load(f, encoding='utf-8')

    params, train_op, loss, feature_map, reconstructed_image = build_ae_model(placeholder_x, config['lr'])
    ae_saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        ae_saver.restore(sess, AE_MODEL_PATH)
        feed_dict = {placeholder_x: x}
        fp, re_image = sess.run([feature_map, reconstructed_image], feed_dict=feed_dict)
        visualize_ae(0, x, fp, re_image)



def main():
    parser = argparse.ArgumentParser(description='COMP5212 Programming Project 2')
    parser.add_argument('--task', default="train", type=str,
                        help='Select the task, train_cnn, test_cnn, '
                             'train_ae, evaluate_ae, ')
    parser.add_argument('--datapath', default="./data", type=str, required=False,
                        help='Select the path to the data directory')
    args = parser.parse_args()
    datapath = args.datapath
    with tf.variable_scope("placeholders"):
        img_var = tf.placeholder(tf.uint8, shape=(None, 28, 28), name="img")
        label_var = tf.placeholder(tf.int32, shape=(None,), name="true_label")

    if args.task == "train_cnn":
        file_train = np.load(datapath + "/data_classifier_train.npz")
        x_train = file_train["x_train"]
        y_train = file_train["y_train"]
        train_cnn(x_train, y_train, img_var, label_var)
    elif args.task == "test_cnn":
        file_test = np.load(datapath + "/data_classifier_test.npz")
        x_test = file_test["x_test"]
        y_test = file_test["y_test"]
        accuracy = test_cnn(x_test, y_test, img_var, label_var)
        print("accuracy = {}\n".format(accuracy))
    elif args.task == "train_ae":
        file_unsupervised = np.load(datapath + "/data_autoencoder_train.npz")
        x_ae_train = file_unsupervised["x_ae_train"]
        train_ae(x_ae_train, img_var)
    elif args.task == "evaluate_ae":
        file_unsupervised = np.load(datapath + "/data_autoencoder_test.npz")
        x_ae_eval = file_unsupervised["x_ae_eval"]
        evaluate_ae(x_ae_eval, img_var)


if __name__ == "__main__":
    main()
