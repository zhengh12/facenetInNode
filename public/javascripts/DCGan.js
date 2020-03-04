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
        model.add(tf.layers.conv2d({filters:256, kernelSize:3, strides:2, padding:'same'}))
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
}

let gan = new DCGan()