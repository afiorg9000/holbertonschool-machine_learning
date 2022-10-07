#!/usr/bin/env python3
"""trains a convolutional NN to classify the CIFAR 10 dataset:"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """trains a convolutional NN to classify the CIFAR 10 dataset:"""
    X_p, Y_p = X, Y
    return (X_p, Y_p)


if __name__ == '__main__':
    train_ds, test_ds = K.datasets.cifar10.load_data()

    inputs = K.layers.Input(shape=(32, 32, 3))

    """USE DenseNet121"""
    tmp_model = K.applications.DenseNet121(
        weights="imagenet",
        input_shape=(128, 128, 3),
        include_top=False,
    )

    lmbda = K.layers.Lambda(lambda x: K.backend.resize_images(x,
                                                              150//32,
                                                              150//32,
                                                              "channels_last"))
    (inputs)

    tmp_model.trainable = False

    x = tmp_model(lmbda, training=False)
    x = K.layers.Flatten()(x)
    x = K.layers.Dense(32, 'relu')(x)
    outputs = K.layers.Dense(10, 'softmax')(x)
    model = K.Model(inputs, outputs)

    model.summary()

    model.compile(
        optimizer=K.optimizers.Adam(),
        loss=K.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[K.metrics.SparseCategoricalAccuracy()],
    )

    def learning_rate_schedule(epoch, lr):
        if epoch < 7:
            return lr
        else:
            return lr * 0.9

    epochs = 10

    model.fit(
        train_ds[0],
        train_ds[1],
        batch_size=32,
        epochs=epochs,
        validation_data=test_ds,
        callbacks=[K.callbacks.LearningRateScheduler(learning_rate_schedule)]
    )

    model.save('cifar10.h5')
