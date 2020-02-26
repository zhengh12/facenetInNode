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