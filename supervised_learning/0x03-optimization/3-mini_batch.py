#!/usr/bin/env python3
"""trains a loaded neural network model using mini-batch gradient descent:"""
import numpy as np
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """loaded neural network model using mini-batch gradient descent:"""
    with tf.Session() as sess:
        # import meta graph and restore session
        new_saver = tf.train.import_meta_graph("{}.meta".format(load_path))
        new_saver.restore(sess, load_path)
        graph = tf.get_default_graph()
        # Get the following tensors and ops from the collection restored
        m = X_train.shape[0]
        x = graph.get_collection('x')[0]
        y = graph.get_collection('y')[0]
        accuracy = graph.get_collection('accuracy')[0]
        loss = graph.get_collection('loss')[0]
        train_op = graph.get_collection('train_op')[0]
        # loop over epochs:
        for epoch in range(epochs + 1):
            steps = m // batch_size + 1
            np.random.permutation(X_train)
            np.random.permutation(Y_train)
            print("After {} epochs:".format(epoch))
            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            print("\tTraining Cost: {}".format(train_cost))
            train_accuracy = sess.run(accuracy, feed_dict={x: X_train,
                                                           y: Y_train})
            print("\tTraining Accuracy: {}".format(train_accuracy))
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            print("\tValidation Cost: {}".format(valid_cost))
            valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid,
                                                           y: Y_valid})
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            x_shuffle, y_shuffle = shuffle_data(X_train, Y_train)
            for step in range(steps):
                start = batch_size * step
                end = batch_size * (step + 1)
                x_batch = x_shuffle[start:end]
                y_batch = y_shuffle[start:end]

                sess.run(
                    train_op,
                    feed_dict={x: x_batch, y: y_batch}
                )
                if (step + 1) % 100 == 0:
                    step_accuracy, step_cost = sess.run(
                        [accuracy, loss], feed_dict={x: x_batch, y: y_batch}
                    )
                    print("\tStep {}:".format(step + 1))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))

        return new_saver.save(sess, save_path)
