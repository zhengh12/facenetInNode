const tf = require("@tensorflow/tfjs-node");
const loadImages = require("./cTreeClassifier");
const facenet = require("./facenet")
const detectFace = require("./detectFace")
const SVM = require('ml-svm');

async function SVM(){
    const trainDatapath = './public/images/RandomForestTrainData1/'
    const predictDatapath = './public/images/RandomForestPredictData/'
    const modelPath = "./public/model/Facenet1/model.json"
    const pModelPath = './public/model/Pnet/model.json'
    const rModelPath = './public/model/Rnet/model.json'
    const oModelPath = './public/model/Onet/model.json'
    const FacenetModel = await facenet.loadFacenetModel(modelPath)
    const mtcnnModel = await detectFace.loadModel(pModelPath, rModelPath, oModelPath)

    var options = {
        C: 0.01,
        tol: 10e-4,
        maxPasses: 10,
        maxIterations: 10000,
        kernel: 'rbf',
        kernelOptions: {
            sigma: 0.5
        }
    }

    let trainVectors = await loadFiles(trainDatapath, FacenetModel, mtcnnModel[0], mtcnnModel[1], mtcnnModel[2])
    let predictVectors = await loadFiles(predictDatapath, FacenetModel, mtcnnModel[0], mtcnnModel[1], mtcnnModel[2])   

    let vectorsAll = []
    let vectorsAvg = []
    for(subVectors of trainVectors){
        let sum = tf.scalar(0)
        for(val of subVectors){
            vectorsAll.push(val)
            sum = sum.add(tf.tensor(val))
        }
        vectorsAvg.push(tf.div(sum,tf.scalar(subVectors.length)).arraySync())
    } 
    let lowVectorsAll = vectorsAll //采用全部向量
    let lowVectors = vectorsAvg //采用类内平均向量
    
} 