#!/usr/bin/env python3
"""builds, trains, and saves a neural network classifier:"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid,
          Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """builds, trains, and saves a neural network classifier:"""
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_prev = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_prev', y_prev)
    loss = calculate_loss(y, y_prev)
    tf.add_to_collection('loss', loss)
    accuracy = calculate_accuracy(y, y_prev)

    tf.add_to_collection('accuracy', accuracy)
    train = create_train_op(loss, alpha)
    tf.add_to_collection('train', train)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(init)

        for i in range(iterations + 1):
            session.run(train, feed_dict={x: X_train, y: Y_train})

            if i in range(iterations + 1):
                acc_train = session.run(accuracy,
                                        feed_dict={x: X_train, y: Y_train})
                acc_valid = session.run(accuracy,
                                        feed_dict={x: X_valid, y: Y_valid})
                cost_train = session.run(loss,
                                         feed_dict={x: X_train, y: Y_train})
                cost_valid = session.run(loss,
                                         feed_dict={x: X_valid, y: Y_valid})

            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(cost_train))
                print("\tTraining Accuracy: {}".format(acc_train))
                print("\tValid Cost: {}".format(cost_valid))
                print("\tValid Accuracy: {}".format(acc_valid))
            if i < iterations:
                session.run(train, feed_dict={x: X_train, y: Y_train})
    return saver.save(session, save_path)
