const tf = require("@tensorflow/tfjs-node");
function generator_model(){
  let model = tf.sequential()
  model.add(tf.layers.dense({inputDim: 100, units: 1024}))
  model.add(tf.layers.activation({activation: "tanh"}))
  model.add(tf.layers.dense({units: 128*7*7}))
  model.add(tf.layers.batchNormalization())
  model.add(tf.layers.activation({activation: "tanh"}))
  model.add(tf.layers.reshape({targetShape: [7, 7, 128], inputShape: [128, 7, 7]}))
  model.add(tf.layers.upSampling2d({size: [2, 2]}))
  model.add(tf.layers.conv2d({filters: 64, kernelSize:[5, 5], padding: 'same'}))
  model.add(tf.layers.activation({activation: "tanh"}))
  model.add(tf.layers.upSampling2d({size: [2, 2]}))
  model.add(tf.layers.conv2d({filters: 1, kernelSize:[5, 5], padding: 'same'}))
  model.add(tf.layers.activation({activation: "tanh"}))
  return model
}

function discriminator_model(){
  let model = tf.sequential()
  model.add(tf.layers.conv2d({filters: 64, kernelSize:[5, 5], padding: 'same', inputShape: [28, 28, 1]}))
  model.add(tf.layers.activation({activation: "tanh"}))
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}))
  model.add(tf.layers.conv2d({filters: 128, kernelSize:[5, 5]}))
  model.add(tf.layers.activation({activation: "tanh"}))
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}))
  model.add(tf.layers.flatten())
  model.add(tf.layers.dense({units: 1024}))
  model.add(tf.layers.activation({activation: "tanh"}))
  model.add(tf.layers.dense({units: 1}))
  model.add(tf.layers.activation({activation: "sigmoid"}))
  return model
}


function generator_containing_discriminator(g, d){
  let model = tf.sequential()
  model.add(g)
  d.trainable = False
  model.add(d)
  return model
}

function combine_images(generated_images){
  num = generated_images.shape[0]
  width = int(math.sqrt(num))
  height = int(math.ceil(float(num)/width))
  shape = generated_images.shape[1:3]
  image = np.zeros((height*shape[0], width*shape[1]),
                    dtype=generated_images.dtype)
  for index, img in enumerate(generated_images):
      i = int(index/width)
      j = index % width
      image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
          img[:, :, 0]
  return image
}


function train(BATCH_SIZE){
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]
    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch)+"_"+str(index)+".png")
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = d.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            d.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
            
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)
}

