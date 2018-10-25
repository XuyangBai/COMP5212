import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.ops.image_ops_impl import ResizeMethod
import time
from Assignment2.common import *


def build_ae_model(placeholder_x, lr, model_num):
    img_float = convert_image_data_to_float(placeholder_x)
    # encoder
    with tf.variable_scope("AE" + str(model_num)) as scope:
        conv1 = tf.layers.conv2d(inputs=img_float,
                                 filters=32,
                                 kernel_size=[3, 3],
                                 strides=(1, 1),
                                 padding='same',
                                 activation=tf.nn.relu,
                                 name='conv1')
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=64,
                                 kernel_size=[3, 3],
                                 strides=(2, 2),
                                 padding='same',
                                 activation=tf.nn.relu,
                                 name='conv2')
        conv3 = tf.layers.conv2d(inputs=conv2,
                                 filters=64,
                                 kernel_size=[3, 3],
                                 strides=(1, 1),
                                 padding='same',
                                 activation=tf.nn.relu,
                                 name='conv3')
    pool2 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding='same', strides=2)
    # feature_map = pool2  # shape [-1, 4, 4, 64]
    feature_map = conv3  # shape [-1,7,7,64]
    # decoder
    depool2 = tf.image.resize_images(images=pool2,
                                     size=[7, 7],
                                     method=ResizeMethod.NEAREST_NEIGHBOR)
    deconv3 = tf.layers.conv2d_transpose(inputs=depool2,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         strides=1,
                                         padding='same',
                                         activation=tf.nn.relu)
    deconv2 = tf.layers.conv2d_transpose(inputs=deconv3,
                                         filters=32,
                                         kernel_size=[3, 3],
                                         strides=2,
                                         padding='same',
                                         activation=tf.nn.relu)
    depool1 = tf.image.resize_images(images=deconv2,
                                     size=[28, 28],
                                     method=ResizeMethod.NEAREST_NEIGHBOR)
    deconv1 = tf.layers.conv2d_transpose(inputs=depool1,
                                         filters=1,
                                         kernel_size=[3, 3],
                                         strides=1,
                                         padding='same',
                                         activation=tf.nn.relu)
    # output = tf.sigmoid(deconv1)
    output = deconv1
    reconstructed_image = tf.cast(output * 255, tf.uint8)

    # loss = tf.losses.mean_squared_error(img_float, reconstructed_image)
    loss = tf.reduce_mean(tf.square(output - img_float))

    params = tf.trainable_variables()
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.9)
    train_op = optimizer.minimize(loss=loss)

    return params, train_op, loss, feature_map, reconstructed_image


def visualize_ae(i, x, features, reconstructed_image):
    plt.figure(0)
    plt.subplot(131)
    plt.imshow(x[i, :, :], cmap="gray")
    plt.title("Origin")
    plt.subplot(132)
    plt.imshow(reconstructed_image[i, :, :, 0], cmap="gray")
    plt.title("Reconstructed")
    plt.subplot(133)
    feature_map = features[i, :, :, 0]
    plt.imshow(feature_map, cmap='gray')
    # plt.imshow(np.reshape(features[i, :, :, :], (7, -1), order="F"), cmap="gray", )
    plt.title("First Feature map")
    plt.show()


def parameter_search(x, placeholder_x):
    return {'lr': 0.001}
    num_epoch = 20
    batch_size = 128
    learning_rate = [0.001, 0.01, 0.1]

    # train validation split for Holdout validation
    train_x = x[0: int(0.8 * x.shape[0])]
    validation_x = x[int(0.8 * x.shape[0]):]

    # Tune the parameters for CAE training with holdout validation
    best_model_loss = 1000000
    best_learning_rate = learning_rate[0]
    model_num = 0
    for lr in learning_rate:
        model_num += 1
        training_set_loss_history = []
        validation_set_loss_history = []
        params, train_op, loss, feature_map, reconstructed_image = build_ae_model(placeholder_x, lr, model_num)
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
                if epoch % 1 == 0:
                    loss_value = sess.run(loss, feed_dict=feed_dict)
                    training_set_loss_history.append(loss_value)
                    loss_value = sess.run(loss, feed_dict={placeholder_x: validation_x})
                    validation_set_loss_history.append(loss_value)
                    print("Epoch {0}, Loss: {1}, Time:{2}s".format(epoch, loss_value, time.time() - start_time))
            # draw the loss over time
            plt.figure()
            plt.plot(np.arange(num_epoch), training_set_loss_history, 'b-', label='train loss')
            plt.plot(np.arange(num_epoch), validation_set_loss_history, 'r-', label='validation loss')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig('./AE/lr={1}.png'.format(AE_MODEL_PATH, lr))
            # finish all the epoch, report the validation loss and compare it with current best
            loss_value, fm, re_image = sess.run([loss, feature_map, reconstructed_image],
                                                feed_dict={placeholder_x: validation_x})
            print("lr={0},batch_size={1}, Validation Loss: {2}, Time: {3}s".format(lr,
                                                                                   batch_size,
                                                                                   loss_value,
                                                                                   time.time() - start_time))

            if loss_value < best_model_loss:
                best_model_loss = loss_value
                best_learning_rate = lr
                print("Update best model with lr {0}, batch size {1}".format(lr, batch_size))
    print("Model with learning_rate {0}, got the best performance, loss value on validation set is {1}".format(
        best_learning_rate,
        best_model_loss))
    config = {}
    config['lr'] = best_learning_rate
    with open('config_ae.pkl', 'wb') as handle:
        pickle.dump(config, handle)


def train_ae(x, placeholder_x):
    config = parameter_search(x, placeholder_x)
    num_epoch = 10
    batch_size = 128

    # use the best hyperparameter  and all the training data to train the model
    start_time = time.time()
    with tf.Session() as sess:
        params, train_op, loss, feature_map, reconstructed_image = build_ae_model(placeholder_x, config['lr'], 0)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        params_save = {
            'conv1': sess.graph.get_tensor_by_name('AE' + str(0) + '/conv1/kernel:0'),
            'conv2': sess.graph.get_tensor_by_name('AE' + str(0) + '/conv2/kernel:0'),
            'conv3': sess.graph.get_tensor_by_name('AE' + str(0) + '/conv3/kernel:0'),
        }
        ae_saver_for_cnn = tf.train.Saver(params_save)
        ae_saver = tf.train.Saver()
        for epoch in range(num_epoch):
            # this time, use all the training set x to train the model with best hyperparameter
            loss_history = []
            num_batch = int(x.shape[0] / batch_size)
            for batch in range(num_batch):
                x_batch = x[batch * batch_size: (batch + 1) * batch_size]
                feed_dict = {placeholder_x: x_batch}
                _ = sess.run([train_op], feed_dict=feed_dict)
            if epoch % 1 == 0:
                loss_value = sess.run(loss, feed_dict=feed_dict)
                loss_history.append(loss_value)
                print("Epoch {0}, Loss: {1}, Time:{2}s".format(epoch, loss_value, time.time() - start_time))
        ae_saver.save(sess, save_path=AE_MODEL_PATH)
        ae_saver_for_cnn.save(sess, save_path=CNN_PRETRAINED_MODEL)
        print("Train the model with best hyperparameter use {} s".format(time.time() - start_time))
        # TODO: very time-consuming
        # loss_value = sess.run(loss, feed_dict={placeholder_x: x})
        # print("The model's loss on training set is {}".format(loss_value))
        plt.figure()
        plt.plot(loss_history, 'g-')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig('./AE/best.png')


def evaluate_ae(x, placeholder_x):
    with open('config_ae.pkl', 'rb') as f:
        config = pickle.load(f, encoding='utf-8')

    params, train_op, loss, feature_map, reconstructed_image = build_ae_model(placeholder_x, config['lr'], 0)
    ae_saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        ae_saver.restore(sess, AE_MODEL_PATH)
        feed_dict = {placeholder_x: x}
        loss_value, fp, re_image = sess.run([loss, feature_map, reconstructed_image], feed_dict=feed_dict)
        print("AE model loss on test set is: {0}".format(loss_value))
        visualize_ae(np.random.randint(0, x.shape[0]), x, fp, re_image)
