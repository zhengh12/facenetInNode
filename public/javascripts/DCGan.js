const tf = require("@tensorflow/tfjs-node");
const mnist = require('mnist-data');
const { cache, trainingImagesUrl, trainingLabelsUrl } = require("mnist-dataset");

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

    randomNumArr(minNum, maxNum, size){ 
        switch(arguments.length){ 
            case 1: 
                return parseInt(Math.random()*minNum+1,10); 
            break; 
            case 2: 
                return parseInt(Math.random()*(maxNum-minNum+1)+minNum,10); 
            break; 
            case 3:
                let randomNumArr = []
                for(let i=0; i<size; i++){
                    randomNumArr.push(parseInt(Math.random()*(maxNum-minNum+1)+minNum,10))
                }
                return randomNumArr
            default: 
                return 0; 
            break; 
        } 
    } 

    train(epochs, batch_size=128, save_interval=50){
        // Load the dataset
        // const trainingImages = await cache(trainingImagesUrl); // [[Number]] (i.e. [[0, 0, 0, ...], [0, 0, 0, ...], ...])
        const dataLength = 60000
        let training_data = mnist.training(dataLength).images.values;
        // console.log(trainingImages[0],trainingImages[0].length)
        console.log(training_data.length, training_data[0].length, training_data[0][0].length) 
        training_data = tf.tensor(training_data)
        console.log(training_data)
        // Rescale -1 to 1
        training_data = tf.div(training_data.sub(tf.scalar(127.5)), tf.scalar(127.5))
        training_data = tf.expandDims(training_data, 3)
        console.log(training_data)

        // Adversarial ground truths
        let valid = tf.ones([batch_size, 1]) 
        let fake = tf.zeros([batch_size, 1])

        for(let i=0; i<epochs; i++){
            // ---------------------
            //  Train Discriminator
            // ---------------------

            // Select a random half of images
            let idx = this.randomNumArr(0, dataLength, batch_size)
            console.log(idx)
            
            return

        }
    }
}

function goit(){
    let gan = new DCGan()
    gan.train(epochs=4000, batch_size=32, save_interval=50)
}

goit()