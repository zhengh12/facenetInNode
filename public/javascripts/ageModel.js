const tf = require("@tensorflow/tfjs-node");
const trainFacenet = require("./trainFacenet");
const config = require("../configParameter/config.json")

async function getModel(){
    let model = await trainFacenet.loadOriginalFacenetModel(config.modelPath.pretrainedAgeModel)
    //model.predict(tf.zeros([1, 299, 299, 3])).print()
    return model
}

exports.getModel = getModel