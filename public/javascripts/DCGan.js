const tf = require("@tensorflow/tfjs-node");

class DCGan {
    constructor() {
        this.img_rows = 28
        this.img_cols = 28
        this.channels = 1
        this.img_shape = [this.img_rows, this.img_cols, this.channels]
        this.latent_dim = 100

        // Build and compile the discriminator
        this.discriminator = this.build_discriminator()
        this.discriminator.summary()
        this.discriminator.compile({loss:'binaryCrossentropy', optimizer: tf.train.adam(0.0002, 0.5), metrics: ['acc']})

        // Build the generator
        this.generator = this.build_generator()

        // The generator takes noise as input and generates imgs
        let z = tf.input({shape:this.latent_dim})
        let img = this.generator.apply(z)

        // For the combined model we will only train the generator
        this.discriminator.trainable = false

        // The discriminator takes generated images as input and determines validity
        let valid = this.discriminator.apply(img)

        // The combined model  (stacked generator and discriminator)
        // Trains the generator to fool the discriminator
        this.combined = tf.model({inputs:z, outputs:valid})
        this.combined.compile({loss:'binaryCrossentropy', optimizer: tf.train.adam(0.0002, 0.5)})
        this.combined.summary()
    }

    build_discriminator(){
        let model = tf.sequential()
        model.add(tf.layers.conv2d({filters:32, kernelSize:3, strides:2, inputShape:this.img_shape, padding:'same'}))
        model.add(tf.layers.leakyReLU({alpha:0.2}))
        model.add(tf.layers.dropout({rate:0.25}))
        model.add(tf.layers.conv2d({filters:64, kernelSize:3, strides:2, padding:'same'}))
        model.add(tf.layers.zeroPadding2d({padding:[[0, 1], [0, 1]]}))
        model.add(tf.layers.batchNormalization({momentum:0.8}))
        model.add(tf.layers.leakyReLU({alpha:0.2}))
        model.add(tf.layers.dropout({rate:0.25}))
        model.add(tf.layers.conv2d({filters:128, kernelSize:3, strides:2, padding:'same'}))
        model.add(tf.layers.batchNormalization({momentum:0.8}))
        model.add(tf.layers.leakyReLU({alpha:0.2}))
        model.add(tf.layers.dropout({rate:0.25}))
        model.add(tf.layers.conv2d({filters:256, kernelSize:3, strides:1, padding:'same'}))
        model.add(tf.layers.batchNormalization({momentum:0.8}))
        model.add(tf.layers.leakyReLU({alpha:0.2}))
        model.add(tf.layers.dropout({rate:0.25}))
        model.add(tf.layers.flatten())
        model.add(tf.layers.dense({units:1, activation:'sigmoid'}))
        model.summary()

        let img = tf.input({shape: this.img_shape})
        let res = model.apply(img)
        return tf.model({inputs:img, outputs:res})
    }

    build_generator(){
        let model = tf.sequential()
        model.add(tf.layers.dense({units:128*7*7, activation:'relu', inputDim:this.latent_dim}))
        model.add(tf.layers.reshape({targetShape:[7, 7, 128]}))
        model.add(tf.layers.upSampling2d({name: 'ys0'}))
        model.add(tf.layers.conv2d({filters:128, kernelSize:3, padding:'same'}))
        model.add(tf.layers.batchNormalization({momentum:0.8}))
        model.add(tf.layers.activation({activation:'relu'}))
        model.add(tf.layers.upSampling2d({name: 'ys1'}))
        model.add(tf.layers.conv2d({filters:64, kernelSize:3, padding:'same'}))
        model.add(tf.layers.batchNormalization({momentum:0.8}))
        model.add(tf.layers.activation({activation:'relu'}))
        model.add(tf.layers.conv2d({filters:this.channels, kernelSize:3, padding:'same'}))
        model.add(tf.layers.activation({activation:'tanh'}))

        model.summary()

        let noise = tf.input({shape: this.latent_dim})
        let res = model.apply(noise)
        return tf.model({inputs:noise, outputs:res})
    }

    train(epochs, batch_size=128, save_interval=50){
        // Load the dataset
        // Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        // Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            // ---------------------
            //  Train Discriminator
            // ---------------------

            // Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            // Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            // Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            // ---------------------
            //  Train Generator
            // ---------------------

            // Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            // Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            // If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
        // Rescale -1 to 1
    }
}

let gan = new DCGan()