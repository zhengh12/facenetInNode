const tf = require("@tensorflow/tfjs-node");
const mnist = require('mnist-data');

function generator_model(){
    let generator = tf.sequential()
    generator.add(tf.layers.dense({units:128*7*7, kernelInitializer:tf.initializers.randomNormal({stddev:0.02}), inputDim:100}))
    generator.add(tf.layers.leakyReLU({alpha:0.2}))
    generator.add(tf.layers.reshape({targetShape:[7, 7, 128]}))
    generator.add(tf.layers.upSampling2d({size:[2,2]}))
    generator.add(tf.layers.conv2d({filters:64, kernelSize:[5,5], padding:'same'}))
    generator.add(tf.layers.leakyReLU({alpha:0.2}))
    generator.add(tf.layers.upSampling2d({size:[2,2]}))
    generator.add(tf.layers.conv2d({filters:1, kernelSize:[5,5], padding:'same', activation:'tanh'}))
    generator.compile({loss: 'binaryCrossentropy', optimizer: 'adam', metrics: ['acc']})
    return generator
}

function discriminator_model(){
    let discriminator = tf.sequential()
    discriminator.add(tf.layers.conv2d({filters:64, kernelSize:[5,5], strides:[2,2], inputShape:[28,28,1], padding:'same', kernelInitializer:tf.initializers.randomNormal({stddev:0.02})}))
    discriminator.add(tf.layers.leakyReLU({alpha:0.2}))
    discriminator.add(tf.layers.dropout({rate:0.3}))
    discriminator.add(tf.layers.conv2d({filters:128, kernelSize:[5,5], strides:[2,2], padding:'same'}))
    discriminator.add(tf.layers.leakyReLU({alpha:0.2}))
    discriminator.add(tf.layers.dropout({rate:0.3}))
    discriminator.add(tf.layers.flatten())
    discriminator.add(tf.layers.dense({units:1, activation:'sigmoid'}))
    discriminator.compile({loss: 'binaryCrossentropy', optimizer: 'adam', metrics: ['acc']})
    return discriminator
}

function generator_containing_discriminator(g, d){
    d.trainable = false
    let ganInput = tf.input({shape: [100]})
    let x = g.apply(ganInput)
    let ganOutput = d.apply(x)
    let gan = tf.model({inputs:ganInput, outputs:ganOutput})
    gan.compile({loss: 'binaryCrossentropy', optimizer: 'adam', metrics: ['acc']})
    return gan 
}

async function train(BATCH_SIZE, X_train){
    let d = discriminator_model()
    console.log("#### discriminator ######")
    d.summary()
    let g = generator_model()
    console.log("#### generator ######")
    g.summary()
    let d_on_g = generator_containing_discriminator(g, d)
    d.trainable = true
    let epoch = 100
    for(let i=0; i<epoch; i++){
        for(let j=0; j<X_train.shape[0]/BATCH_SIZE; j++){
            let noise = tf.randomUniform([BATCH_SIZE, 100], 0, 1)
            let image_batch = X_train.slice([j*BATCH_SIZE, 0, 0, 0], [(j+1)*BATCH_SIZE, 28, 28, 1])
            console.log(image_batch)
            // image_batch.print()
            let generated_images = g.predict(noise,{verbose:0})
            let X = tf.concat([image_batch, generated_images])
            console.log(X.shape)
            let y = tf.concat([tf.ones([BATCH_SIZE]), tf.zeros([BATCH_SIZE])])
            console.log(y.shape)
            let d_loss = await d.trainOnBatch(X, y)
            console.log(d_loss)
            noise = tf.randomUniform([BATCH_SIZE, 100], 0, 1)
            d.trainable = false
            let g_loss = await d_on_g.trainOnBatch(noise, tf.ones([BATCH_SIZE]))
            d.trainable = true
            console.log(g_loss)
            return 
        }
    }
}

async function run(){
    const dataLength = 60000
    let training_data = mnist.training(dataLength).images.values;
    training_data = tf.tensor(training_data)
    training_data = tf.div(training_data.sub(tf.scalar(127.5)), tf.scalar(127.5))
    training_data = tf.expandDims(training_data, 3)
    console.log(training_data)
    let [Model_d, Model_g] = await train(32, training_data)
}
run()
