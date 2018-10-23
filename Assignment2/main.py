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
CNN_MODEL_PATH = "./CNN/cnn_model"
AE_MODEL_PATH = "./AE/ae_model"


# Following functions are helper functions that you can feel free to change
def convert_image_data_to_float(image_raw):
    img_float = tf.expand_dims(tf.cast(image_raw, tf.float32) / 255, axis=-1)
    return img_float


def build_cnn_model(placeholder_x, placeholder_y, H, lr):
    img_float = convert_image_data_to_float(placeholder_x)
    conv1 = tf.layers.conv2d(inputs=img_float,
                             filters=32,
                             kernel_size=[3, 3],
                             strides=(1, 1),
                             padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2, 2],
                                    strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[3, 3],
                             strides=(2, 2),
                             padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
    conv3 = tf.layers.conv2d(inputs=conv2,
                             filters=64,
                             kernel_size=[3, 3],
                             strides=(1, 1),
                             padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool2 = tf.layers.max_pooling2d(inputs=conv3,
                                    pool_size=[2, 2],
                                    padding='same',
                                    strides=2)
    fc1 = tf.contrib.layers.fully_connected(inputs=tf.reshape(pool2, [-1, np.prod(pool2.shape[1:])]),
                                            num_outputs=H,
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer())
    fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
                                            num_outputs=47,
                                            activation_fn=tf.nn.sigmoid,
                                            weights_initializer=tf.contrib.layers.xavier_initializer())
    logits = fc2
    # sparse_softmax_cross_entropy_with_logits return the same shape with labels
    loss = tf.losses.sparse_softmax_cross_entropy(labels=placeholder_y, logits=logits)

    # compute the accuracy
    y = tf.one_hot(placeholder_y, NUM_LABELS)
    correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999)
    train_op = optimizer.minimize(loss)

    return train_op, loss, accuracy, tf.trainable_variables()


# Major interfaces
def train_cnn(x, y, placeholder_x, placeholder_y):
    num_iterations = 20
    batch_size = 128
    # Hs = [512, 1024]
    Hs = [512, 1024]
    learning_rates = [0.001, 0.01, 0.1]
    holdout = [0.5]
    # learning_rates = [0.001, 0.01, 0.1]
    # holdout = [0.5, 1]
    best_model_acc = 0
    config = {'H': Hs[0], 'lr': learning_rates[0], 'ratio': holdout[0]}
    for H in Hs:
        for lr in learning_rates:
            for ratio in holdout:
                train_op, loss, accuracy, params = build_cnn_model(placeholder_x, placeholder_y, H, lr)
                cnn_saver = tf.train.Saver(max_to_keep=None)
                start_time = time.time()
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    x_train = x[0:int(x.shape[0] * ratio)]
                    y_train = y[0:int(x.shape[0] * ratio)]
                    x_validation = x[int(x.shape[0] * ratio):]
                    y_validation = y[int(x.shape[0] * ratio):]
                    for epoch in range(num_iterations):
                        for batch in range(int(x_train.shape[0] / batch_size)):
                            x_batch = x_train[batch * batch_size: (batch + 1) * batch_size]
                            y_batch = y_train[batch * batch_size: (batch + 1) * batch_size]
                            feed_dict = {placeholder_x: x_batch, placeholder_y: y_batch}
                            _ = sess.run([train_op], feed_dict=feed_dict)
                            # if batch % 10 == 0:
                            #     loss_value = sess.run([loss], feed_dict=feed_dict)
                            #     print("Epoch{0}, miniBatch{1}, Loss:{2}".format(epoch, batch, loss_value))
                        loss_value, acc_value = sess.run([loss, accuracy], feed_dict=feed_dict)
                        print("Epoch{0} End, Loss:{1}, Accuracy:{2}, Time:{3}".format(epoch, loss_value, acc_value,
                                                                                      time.time() - start_time))
                    loss_value, acc_value = sess.run([loss, accuracy], feed_dict={placeholder_x: x_validation,
                                                                                  placeholder_y: y_validation})
                    print("H={0},lr={1},ratio={2} Validation Loss={3}, Accuracy{4}, Time={5}".format(H, lr, ratio,
                                                                                                     loss_value,
                                                                                                     acc_value,
                                                                                                     time.time() - start_time))
                    if acc_value > best_model_acc:
                        best_model_acc = acc_value
                        config['H'] = H
                        config['lr'] = lr
                        config['ratio'] = ratio
                        cnn_saver.save(sess, save_path=CNN_MODEL_PATH)
    print("Model with lr={0}, H={1}, ratio={2} got the best performance, Accuracy is {3}".format(config['lr'],
                                                                                                 config['H'],
                                                                                                 config['ratio'],
                                                                                                 best_model_acc))
    with open('config_cnn.pkl', 'wb') as f:
        pickle.dump(config, f)


def test_cnn(x, y, placeholder_x, placeholder_y):
    with open('config_cnn.pkl', 'rb') as f:
        config = pickle.load(f, encoding='utf-8')

    train_op, loss, accuracy, params = build_cnn_model(placeholder_x, placeholder_y, config['H'], config['lr'])
    cnn_saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        cnn_saver.restore(sess, CNN_MODEL_PATH)
        feed_dict = {placeholder_x: x, placeholder_y: y}
        result_accuracy = sess.run(accuracy, feed_dict=feed_dict)
        print("Holdout validation with p = 0:5: Accuracy on test set:{}%".format(result_accuracy * 100))
    return result_accuracy


def build_ae_model(placeholder_x, lr):
    img_float = convert_image_data_to_float(placeholder_x)
    # encoder
    conv1 = tf.layers.conv2d(inputs=img_float, filters=32, kernel_size=[3, 3], strides=(1, 1), padding='same',
                             activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], strides=(2, 2), padding='same',
                             activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3], strides=(1, 1), padding='same',
                             activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding='same', strides=2)
    # feature_map = pool2  # shape [-1, 4, 4, 64]
    feature_map = conv3  # shape [-1,7,7,64]
    # decoder
    depool2 = tf.image.resize_images(images=pool2, size=[7, 7], method=ResizeMethod.NEAREST_NEIGHBOR)
    deconv3 = tf.layers.conv2d_transpose(inputs=depool2, filters=64, kernel_size=[3, 3], strides=1, padding='same',
                                         activation=tf.nn.relu)
    deconv2 = tf.layers.conv2d_transpose(inputs=deconv3, filters=32, kernel_size=[3, 3], strides=2, padding='same',
                                         activation=tf.nn.relu)
    depool1 = tf.image.resize_images(images=deconv2, size=[28, 28], method=ResizeMethod.NEAREST_NEIGHBOR)
    deconv1 = tf.layers.conv2d_transpose(inputs=depool1, filters=1, kernel_size=[3, 3], strides=1, padding='same',
                                         activation=tf.nn.relu)

    reconstructed_image = tf.expand_dims(tf.cast(deconv1 * 255, tf.int32), axis=-1)
    # reconstructed_image = deconv1
    # loss = tf.losses.mean_squared_error(img_float, reconstructed_image)
    loss = tf.reduce_mean(tf.square(reconstructed_image - img_float))
    # print([x.name for x in tf.trainable_variables()])

    params = tf.trainable_variables()
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.9)
    train_op = optimizer.minimize(loss=loss)

    return params, train_op, loss, feature_map, reconstructed_image


def visualize_ae(i, x, features, reconstructed_image):
    plt.figure(0)
    plt.imshow(x[i, :, :], cmap="gray")
    plt.figure(1)
    plt.imshow(reconstructed_image[i, :, :, 0], cmap="gray")
    plt.figure(2)
    plt.imshow(np.reshape(features[i, :, :, :], (7, -1), order="F"), cmap="gray", )
    plt.show()


def train_ae(x, placeholder_x):
    num_epoch = 10
    # learning_rate = [0.1, 0.01, 0.001]
    learning_rate = [0.001]
    batch_sizes = [128]
    # batch_sizes = [64 128, 256]

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
                        print("Epoch {0}, Loss: {1}, Time:{2}s".format(epoch, loss_value, time.time() - start_time))
                # finish all the epoth, report the validation loss and compare it with current best
                loss_value = sess.run([loss], feed_dict={placeholder_x: validation_x})
                print("lr={0},batch_size={1}, Validation Loss: {2}, Time: {3}s".format(lr, batch_size, loss_value,
                                                                                       time.time() - start_time))
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
