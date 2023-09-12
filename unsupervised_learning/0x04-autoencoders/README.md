# 0x04 Autoencoders

> An **autoencoder** is a type of artificial neural network used to learn efficient codings of unlabeled data (unsupervised learning). The encoding is validated and refined by attempting to regenerate the input from the encoding. The autoencoder learns a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore insignificant data (“noise”)

At the end of this project I was able to answer these conceptual questions:

* What is an autoencoder?
* What is latent space?
* What is a bottleneck?
* What is a sparse autoencoder?
* What is a convolutional autoencoder?
* What is a generative model?
* What is a variational autoencoder?
* What is the Kullback-Leibler divergence?

## Tasks

0. Write a function `def autoencoder(input_dims, hidden_layers, latent_dims):` that creates an autoencoder:

    * `input_dims` is an integer containing the dimensions of the model input
    * `hidden_layers` is a list containing the number of nodes for each hidden layer in the encoder, respectively
        * the hidden layers should be reversed for the decoder
    * `latent_dims` is an integer containing the dimensions of the latent space representation
    * Returns: `encoder, decoder, auto`
        * `encoder` is the encoder model
        * `decoder` is the decoder model
        * `auto` is the full autoencoder model
    * The autoencoder model should be compiled using adam optimization and binary cross-entropy loss
    * All layers should use a `relu` activation except for the last layer in the decoder, which should use `sigmoid`

1. Write a function `def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):` that creates a sparse autoencoder:

    * `input_dims` is an integer containing the dimensions of the model input
    * `hidden_layers` is a list containing the number of nodes for each hidden layer in the encoder, respectively
        * the hidden layers should be reversed for the decoder
    * `latent_dims` is an integer containing the dimensions of the latent space representation
    * `lambtha` is the regularization parameter used for L1 regularization on the encoded output
    * Returns: `encoder, decoder, auto`
        * `encoder` is the encoder model
        * `decoder` is the decoder model
        * `auto` is the sparse autoencoder model
    * The sparse autoencoder model should be compiled using adam optimization and binary cross-entropy loss
    * All layers should use a `relu` activation except for the last layer in the decoder, which should use `sigmoid`

2. Write a function `def autoencoder(input_dims, filters, latent_dims):` that creates a convolutional autoencoder:

    * `input_dims` is a tuple of integers containing the dimensions of the model input
    * `filters` is a list containing the number of filters for each convolutional layer in the encoder, respectively
        * the filters should be reversed for the decoder
    * `latent_dims` is a tuple of integers containing the dimensions of the latent space representation
    * Each convolution in the encoder should use a kernel size of `(3, 3)` with same padding and `relu` activation, followed by max pooling of size `(2, 2)`
    * Each convolution in the decoder, except for the last two, should use a filter size of `(3, 3)` with same padding and `relu` activation, followed by upsampling of size `(2, 2)`
        * The second to last convolution should instead use valid padding
        * The last convolution should have the same number of filters as the number of channels in `input_dims` with `sigmoid` activation and no upsampling
    * Returns: `encoder, decoder, auto`
        * `encoder` is the encoder model
        * `decoder` is the decoder model
        * `auto` is the full autoencoder model
    * The autoencoder model should be compiled using adam optimization and binary cross-entropy loss

3. Write a function `def autoencoder(input_dims, hidden_layers, latent_dims):` that creates a variational autoencoder:

    * `input_dims` is an integer containing the dimensions of the model input
    * `hidden_layers` is a list containing the number of nodes for each hidden layer in the encoder, respectively
        * the hidden layers should be reversed for the decoder
    * `latent_dims` is an integer containing the dimensions of the latent space representation
    * Returns: `encoder, decoder, auto`
        * `encoder` is the encoder model, which should output the latent representation, the mean, and the log variance, respectively
        * `decoder` is the decoder model
        * `auto` is the full autoencoder model
    * The autoencoder model should be compiled using adam optimization and binary cross-entropy loss
    * All layers should use a `relu` activation except for the mean and log variance layers in the encoder, which should use `None`, and the last layer in the decoder, which should use `sigmoid`

