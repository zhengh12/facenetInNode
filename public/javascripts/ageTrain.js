const tf = require("@tensorflow/tfjs-node");
const ageModel = require("./ageModel");
const generator = require("./ageDataGenerator")
const config = require("../configParameter/config.json")

async function trainAge(){
    let originalModel = await ageModel.getModel()
    originalModel.compile({optimizer:tf.train.sgd(1e-5), loss:"categoricalCrossentropy", metrics: ['acc']})
    console.log(originalModel.summary())
    let hist = originalModel.fitDataset(await generator.GetData("train"), {
        batchesPerEpoch: Math.ceil(4113 / config.trainParam.batch_size),
        epochs: config.trainParam.epochs,
        verbose: 1,
        callbacks: {
            onTrainBegin: async () => {
                console.log("onTrainBegin")
              },
              onTrainEnd: async (epoch, logs) => {
                console.log("onTrainEnd" + epoch + JSON.stringify(logs))
              },
              onEpochBegin: async (epoch, logs) => {
                console.log("onEpochBegin" + epoch + JSON.stringify(logs))
              },
              onEpochEnd: async (epoch, logs) => {
                console.log("onEpochEnd" + epoch + JSON.stringify(logs))
              },
              onBatchBegin: async (epoch, logs) => {
                console.log("onBatchBegin" + epoch + JSON.stringify(logs))
              },
              onBatchEnd: async (epoch, logs) => {
                console.log("onBatchEnd" + epoch + JSON.stringify(logs))
              }
        },
        validationData: await generator.GetData("validation"),
        validationBatches: Math.ceil(1500 / config.trainParam.batch_size)
    })
}
trainAge()