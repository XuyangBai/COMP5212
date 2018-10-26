import pickle
import tensorflow as tf
import numpy as np
from Assignment2.common import *
import tensorflow as tf
import matplotlib.pyplot as plt
import time


def build_cnn_model(placeholder_x, placeholder_y, H, lr, model_num):
    img_float = convert_image_data_to_float(placeholder_x)
    with tf.variable_scope("CNN" + str(model_num)) as scope:
        conv1 = tf.layers.conv2d(inputs=img_float,
                                 filters=32,
                                 kernel_size=[3, 3],
                                 strides=(1, 1),
                                 padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name='conv1')
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=64,
                                 kernel_size=[3, 3],
                                 strides=(2, 2),
                                 padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name='conv2')
        conv3 = tf.layers.conv2d(inputs=conv2,
                                 filters=64,
                                 kernel_size=[3, 3],
                                 strides=(1, 1),
                                 padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name='conv3')
        pool2 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding='same', strides=2)
        fc1 = tf.contrib.layers.fully_connected(inputs=tf.reshape(pool2, [-1, np.prod(pool2.shape[1:])]),
                                                num_outputs=H,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
        fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
                                                num_outputs=47,
                                                activation_fn=tf.nn.sigmoid,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
    logits = fc2
    loss = tf.losses.sparse_softmax_cross_entropy(labels=placeholder_y, logits=logits)
    # compute the accuracy
    y = tf.one_hot(placeholder_y, NUM_LABELS)
    correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999)
    train_op = optimizer.minimize(loss)

    return train_op, tf.log(loss), accuracy, tf.trainable_variables()


def parameter_search(x, y, placeholder_x, placeholder_y, pretrained):
    # return {'H': 1024, 'lr': 0.001}
    num_epoch = 50
    batch_size = 128
    Hs = [512, 1024]
    learning_rates = [0.001, 0.01, 0.1]

    # train validation split for holdout validation
    x_train = x[0:int(x.shape[0] * 0.8)]
    y_train = y[0:int(x.shape[0] * 0.8)]
    x_validation = x[int(x.shape[0] * 0.8):]
    y_validation = y[int(x.shape[0] * 0.8):]

    best_model_acc = 0
    config = {'H': Hs[0], 'lr': learning_rates[0]}
    model_num = 0
    for H in Hs:
        for lr in learning_rates:
            training_set_loss = []
            validation_set_loss = []
            training_set_accuracy = []
            validation_set_accuracy = []
            model_num += 1
            train_op, loss, accuracy, params = build_cnn_model(placeholder_x, placeholder_y, H, lr, model_num)

            start_time = time.time()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                # if use pretrained model, load the saved checkpoint first
                if pretrained is True:
                    params_save = {
                        'conv1': sess.graph.get_tensor_by_name('CNN' + str(model_num) + '/conv1/kernel:0'),
                        'conv2': sess.graph.get_tensor_by_name('CNN' + str(model_num) + '/conv2/kernel:0'),
                        'conv3': sess.graph.get_tensor_by_name('CNN' + str(model_num) + '/conv3/kernel:0'),
                    }
                    cnn_saver = tf.train.Saver(params_save)
                    cnn_saver.restore(sess, CNN_PRETRAINED_MODEL)
                for epoch in range(num_epoch):
                    for batch in range(int(x_train.shape[0] / batch_size)):
                        x_batch = x_train[batch * batch_size: (batch + 1) * batch_size]
                        y_batch = y_train[batch * batch_size: (batch + 1) * batch_size]
                        feed_dict = {placeholder_x: x_batch, placeholder_y: y_batch}
                        _ = sess.run([train_op], feed_dict=feed_dict)
                    # every epoch, save the loss and accuracy for plot.
                    loss_value, acc_value = sess.run([loss, accuracy], feed_dict=feed_dict)
                    training_set_accuracy.append(acc_value)
                    training_set_loss.append(loss_value)
                    loss_value, acc_value = sess.run([loss, accuracy], feed_dict={placeholder_x: x_validation,
                                                                                  placeholder_y: y_validation})
                    validation_set_accuracy.append(acc_value)
                    validation_set_loss.append(loss_value)
                    print("Epoch{0}, Loss:{1:.3f}, Accuracy:{2:.3f}, Time:{3:.3f}".format(epoch, loss_value, acc_value,
                                                                                          time.time() - start_time))
                loss_value, acc_value = sess.run([loss, accuracy], feed_dict={placeholder_x: x_validation,
                                                                              placeholder_y: y_validation})
                print("H={0},lr={1}, Validation Loss={2:.3f}, Accuracy:{3:.3f}, Time={4:.3f}".format(H, lr, loss_value,
                                                                                                     acc_value,
                                                                                                     time.time() - start_time))
                plt.figure()
                plt.plot(np.arange(num_epoch), training_set_accuracy, 'g', label='training accuracy')
                plt.plot(np.arange(num_epoch), validation_set_accuracy, 'b', label='validation accuracy')
                plt.legend()
                plt.xlabel('epoch')
                plt.ylabel('accuracy')
                if pretrained:
                    plt.savefig('./CNN_pretrained/accuracy/lr={0},H={1}.png'.format(lr, H))
                else:
                    plt.savefig('./CNN/accuracy/lr={0},H={1}.png'.format(lr, H))
                plt.figure()
                plt.plot(np.arange(num_epoch), training_set_loss, 'g', label='training loss')
                plt.plot(np.arange(num_epoch), validation_set_loss, 'b', label='validation loss')
                plt.legend()
                plt.xlabel('epoch')
                plt.ylabel('loss')
                if pretrained:
                    plt.savefig('./CNN_pretrained/loss/lr={0},H={1}.png'.format(lr, H))
                else:
                    plt.savefig('./CNN/loss/lr={0},H={1}.png'.format(lr, H))
                if acc_value > best_model_acc:
                    best_model_acc = acc_value
                    config['H'] = H
                    config['lr'] = lr
    print("Model with lr={0}, H={1}, got the best performance, Accuracy is {2:.3f}".format(config['lr'],
                                                                                           config['H'],
                                                                                           best_model_acc))
    with open('config_cnn.pkl', 'wb') as f:
        pickle.dump(config, f)
    return config


def train_cnn(x, y, placeholder_x, placeholder_y, pretrained=True):

    config = parameter_search(x, y, placeholder_x, placeholder_y, pretrained)

    num_epoch = 50
    batch_size = 128

    # use the best hyperparameter and all the training data to train the model
    start_time = time.time()
    training_set_loss = []
    training_set_accuracy = []
    with tf.Session() as sess:
        # model_num set to 0 for reuse
        train_op, loss, accuracy, params = build_cnn_model(placeholder_x, placeholder_y, config['H'], config['lr'], 0)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        cnn_saver = tf.train.Saver()
        if pretrained is True:
            params_save = {
                'conv1': sess.graph.get_tensor_by_name('CNN' + str(0) + '/conv1/kernel:0'),
                'conv2': sess.graph.get_tensor_by_name('CNN' + str(0) + '/conv2/kernel:0'),
                'conv3': sess.graph.get_tensor_by_name('CNN' + str(0) + '/conv3/kernel:0'),
            }
            cnn_saver_pretrained = tf.train.Saver(params_save)
            cnn_saver_pretrained.restore(sess, CNN_PRETRAINED_MODEL)
        for epoch in range(num_epoch):
            for batch in range(int(x.shape[0] / batch_size)):
                x_batch = x[batch * batch_size: (batch + 1) * batch_size]
                y_batch = y[batch * batch_size: (batch + 1) * batch_size]
                feed_dict = {placeholder_x: x_batch, placeholder_y: y_batch}
                _ = sess.run([train_op], feed_dict=feed_dict)
            # every epoch, save the loss and accuracy for plot.
            loss_value, acc_value = sess.run([loss, accuracy], feed_dict=feed_dict)
            training_set_accuracy.append(acc_value)
            training_set_loss.append(loss_value)
            print("Epoch{0} End, Loss:{1:.3f}, Accuracy:{2:.3f}, Time:{3:.3f}".format(epoch, loss_value, acc_value,
                                                                                      time.time() - start_time))

        cnn_saver.save(sess, save_path=CNN_MODEL_PATH)
        loss_value, acc_value = sess.run([loss, accuracy], feed_dict={placeholder_x: x,
                                                                      placeholder_y: y})
        print("Train the model with best hyerparameter use {:.1f}s".format(time.time() - start_time))
        print("The model's loss on the training set is {:.3f}".format(loss_value))
        print("The model's accuracy on the training set is {:.1f}%".format(acc_value * 100))
        plt.figure()
        plt.subplot(121)
        plt.plot(np.arange(num_epoch), training_set_accuracy, 'g', label='training accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.subplot(122)
        plt.plot(np.arange(num_epoch), training_set_loss, 'b', label='training loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        if pretrained:
            plt.savefig('./CNN_pretrained/best.png')
        else:
            plt.savefig('./CNN/best.png')


def test_cnn(x, y, placeholder_x, placeholder_y, pretrained=True):
    with open('config_cnn.pkl', 'rb') as f:
        config = pickle.load(f, encoding='utf-8')
    # model_num = 0 means the model with best parameters
    train_op, loss, accuracy, params = build_cnn_model(placeholder_x, placeholder_y, config['H'], config['lr'], 0)
    cnn_saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        cnn_saver.restore(sess, CNN_MODEL_PATH)
        feed_dict = {placeholder_x: x, placeholder_y: y}
        loss_value, result_accuracy = sess.run([loss, accuracy], feed_dict=feed_dict)
        print("Loss on test set:{}".format(loss_value))
    return result_accuracy
