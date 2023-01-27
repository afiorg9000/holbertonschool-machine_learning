#!/usr/bin/env python3
"""creates and trains a transformer mode"""
import tensorflow as tf
import tensorflow.keras as K
import numpy as np

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """creates and trains a transformer mode"""
    data = Dataset(batch_size, max_len)
    train_data = data.data_train
    val_data = data.data_valid
    steps_per_epoch = data.data_train[0].shape[0] // batch_size
    vocab_size = data.tokenizer_pt.vocab_size + 2
    dataset = tf.data.Dataset.from_generator(data.data_generator,
                                             output_types=(tf.int64, tf.int64),
                                             output_shapes=(tf.TensorShape([None]),
                                                            tf.TensorShape([None])))
    dataset = dataset.cache()
    dataset = dataset.shuffle(data.data_train[0].shape[0])
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=([None], [None]))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    transformer = Transformer(N, dm, h, hidden, vocab_size, vocab_size)
    learning_rate = CustomSchedule(dm)
    optimizer = K.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                  epsilon=1e-9)
    loss_object = K.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    train_loss = K.metrics.Mean(name='train_loss')
    train_accuracy = K.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    val_loss = K.metrics.Mean(name='val_loss')
    val_accuracy = K.metrics.SparseCategoricalAccuracy(
        name='val_accuracy')

    @tf.function
    def train_step(inp, tar):
        """creates and trains a transformer mode"""
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp,
                                                                         tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
            loss = loss_function(tar_real, predictions)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients,
                                      transformer.trainable_variables))
        train_loss(loss)
        train_accuracy(tar_real, predictions)

    @tf.function
    def val_step(inp, tar):
        """creates and trains a transformer mode"""
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp,
                                                                         tar_inp)
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)
        val_loss(loss)
        val_accuracy(tar_real, predictions)

    def loss_function(real, pred):
        """creates and trains a transformer mode"""
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    for epoch in range(epochs):
        """creates and trains a transformer mode"""
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()
        for (batch, (inp, tar)) in enumerate(dataset):
            train_step(inp, tar)
            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(),
                    train_accuracy.result()))
        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))
        print('Validation Loss {:.4f} Accuracy {:.4f}'.format(val_loss.result(),
                                                              val_accuracy.result()))
    return transformer


class CustomSchedule(K.optimizers.schedules.LearningRateSchedule):
    """creates and trains a transformer mode"""
    def __init__(self, d_model, warmup_steps=4000):
        """creates and trains a transformer mode"""
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """creates and trains a transformer mode"""
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
