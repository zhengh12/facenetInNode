const tf = require("@tensorflow/tfjs-node");

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
    generator.compile({loss: 'binaryCrossentropy', optimizer: 'adam'})
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
    discriminator.compile({loss: 'binaryCrossentropy', optimizer: 'adam'})
    return discriminator
}

function generator_containing_discriminator(g, d){
    d.trainable = false
    let ganInput = tf.input({shape: [100]})
    let x = g.apply(ganInput)
    let ganOutput = d.apply(x)
    let gan = tf.model({inputs:ganInput, outputs:ganOutput})
    gan.compile({loss: 'binaryCrossentropy', optimizer: 'adam'})
    return gan 
}

function train(BATCH_SIZE, X_train){
    let d = discriminator_model()
    console.log("#### discriminator ######")
    d.summary()
    let g = generator_model()
    console.log("#### generator ######")
    g.summary()
    let d_on_g = generator_containing_discriminator(g, d)
    d.trainable = true
}

function compute_anomaly_score(model, x){
    z = np.random.uniform(0, 1, size=(1, 100))
    intermidiate_model = feature_extractor()
    d_x = intermidiate_model.predict(x)
    loss = model.fit(z, [x, d_x], epochs=500, verbose=0)
    similar_data, _ = model.predict(z)
    return loss.history['loss'][-1], similar_data
}
train()
