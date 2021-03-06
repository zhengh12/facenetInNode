const tf = require("@tensorflow/tfjs-node");
const facenet = require("./facenet")
const detectFace = require("./detectFace")
const SVM = require('ml-svm');
const ProgressBar = require('progress');
const fs = require("fs");
//读到文件夹下所有文件的路径
function findFile(dataPath){
    let dirArr = []
    let total = 0
    let dir = fs.readdirSync(dataPath)
    dir.map(item=>{
        let stat = fs.lstatSync(dataPath + item)
        if (stat.isDirectory() === true) { 
            let subDirArr = []
            let subDir = fs.readdirSync(dataPath + item + '/')
            subDir.map(val=>{
                let stat = fs.lstatSync(dataPath + item + '/' + val)
                if(stat.isDirectory() === false){
                    subDirArr.push(dataPath + item + '/' + val)
                    total += 1
                }
            })
            dirArr.push(subDirArr)
        }
    })
    return [dirArr, total]
}

//主文件夹下包含多个子文件夹，每个子文件夹作为一个人脸分类，一个子文件夹包含至少一张人脸图片
async function loadFiles(dataPath, FacenetModel, Pnet, Rnet, Onet){
    let [dirArr, total] = findFile(dataPath)
    const bar = new ProgressBar('  loading :filename Progress :bar :rate/nps :percent :etas', {
        complete: '█',
        incomplete: '░',
        width: 20,
        total: total
    });
    let vectors = []
    for(let i=0; i<dirArr.length; i++){
        let subvector = []
        for(val of dirArr[i]){
            bar.tick({
                'filename': val.split("/").pop()
            });
            let vector = await facenet.faceAlignVector(FacenetModel, val, Pnet, Rnet, Onet)
            vector = vector.arraySync()[0]
            // vector.push(i)
            // vector = vector.map((val,index)=>{
            //     return vector.slice(0,index).concat(vector.slice(index+1,vector.length)).concat([index.toString()])
            // })
            subvector.push(vector)
        }
        vectors.push(subvector)
    }
    console.log("loading file over")
    return vectors
}

function createTree(trainVectors, predictVectors, SVMTree, index){
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
    let convectorsAvg = vectorsAvg //采用类内平均向量
    
    let furthest = []
    let max = 0
    for(let m=0; m<convectorsAvg.length; m++){
        for(let n=0; n<convectorsAvg.length; n++){
            let dis = tf.sqrt(tf.sum(tf.squaredDifference(tf.tensor(convectorsAvg[m]),tf.tensor(convectorsAvg[n])))).arraySync()
            furthest = dis>max ? [m,n] : furthest
            max = dis>max ? dis : max
        }
    }
    console.log(furthest)
    //let deletenum = trainVector[furthest[0]]
    // let class1 = trainVectors.slice(0, furthest[0]).concat(trainVectors.slice(furthest[0]+1, trainVectors.length))
    // let class2 = trainVectors.slice(0, furthest[1]).concat(trainVectors.slice(furthest[1]+1, trainVectors.length))
    // let class1All = []
    // let class2All = []
    // for(subVectors of class1){
    //     for(val of subVectors){
    //         class1All.push(val)
    //     }
    // } 
    // for(subVectors of class2){
    //     for(val of subVectors){
    //         class2All.push(val)
    //     }
    // } 
    // console.log(class1All.length, class2All.length)
    // let len = class1All.length
    // trainVectors = class1All.concat(class2All)
    let choiceTrainVectors = trainVectors[furthest[0]].concat(trainVectors[furthest[1]])
    let predictions = []
    for(let i=0; i<trainVectors[furthest[0]].length; i++){
        predictions.push(1)
    }
    for(let i=0; i<trainVectors[furthest[1]].length; i++){
        predictions.push(-1)
    }
    // for(let i=0; i<trainVectors.length; i++){
    //     for(let j=0; j<trainVectors[i].length; j++){
    //         if(i==0){
    //             predictions.push(1)
    //         }else{
    //             predictions.push(-1)
    //         }
    //     }
    // }
    var options = {
        C: 0.1,
        tol: 10e-4,
        maxPasses: 10,
        maxIterations: 10000,
        kernel: 'linear',
    }
    var svm = new SVM(options)
    svm.train(choiceTrainVectors, predictions)
    SVMTree.model = svm.toJSON()
    console.log(trainVectors.length, predictVectors.length)
    console.log(svm.predict(trainVectors[furthest[0]]),svm.predict(trainVectors[furthest[1]]))
    console.log(svm.predict(predictVectors[1]))
    // console.log(svm.predict(predictVectors[furthest[1]]))
    // offset0 = del<=furthest[0]?offset0+1:offset0
    // offset1 = del<=furthest[1]?offset1+1:offset1
    // console.log("del:",del, offset0, offset1)
    // SVMTree.leftC = {target: furthest[0]+offset0}
    // SVMTree.rightC = {target: furthest[1]+offset1}
    SVMTree.leftC = {target: index[furthest[0]]}
    SVMTree.rightC = {target: index[furthest[1]]}
    if(trainVectors.length>2){
        console.log(trainVectors.slice(0, furthest[0]).concat(trainVectors.slice(furthest[0]+1, trainVectors.length)).length)
        createTree(trainVectors.slice(0, furthest[0]).concat(trainVectors.slice(furthest[0]+1, trainVectors.length)), predictVectors.slice(0, furthest[0]).concat(predictVectors.slice(furthest[0]+1, predictVectors.length)), SVMTree.leftC, index.slice(0, furthest[0]).concat(index.slice(furthest[0]+1, index.length)))
        createTree(trainVectors.slice(0, furthest[1]).concat(trainVectors.slice(furthest[1]+1, trainVectors.length)), predictVectors.slice(0, furthest[1]).concat(predictVectors.slice(furthest[1]+1, predictVectors.length)), SVMTree.rightC, index.slice(0, furthest[1]).concat(index.slice(furthest[1]+1, index.length)))
    }
}

function TreePredict(SVMTree, toPredict){
    let scanNode = SVMTree
    while(scanNode.leftC.model != null){
        let importedSvm = SVM.load(scanNode.model)
        console.log(scanNode.target)
        if(importedSvm.predict(toPredict)==1){
            scanNode = scanNode.rightC
        }else{
            scanNode = scanNode.leftC
        }
    }
    // console.log(scanNode)
    let resultSVM = SVM.load(scanNode.model)
    let target = resultSVM.predict(toPredict)==1 ? scanNode.leftC.target : scanNode.rightC.target
    console.log("final:",target)
}

async function SVMClassifier(){
    const trainDatapath = './public/images/RandomForestTrainData/'
    const predictDatapath = './public/images/RandomForestPredictData/'
    const modelPath = "./public/model/Facenet1/model.json"
    const pModelPath = './public/model/Pnet/model.json'
    const rModelPath = './public/model/Rnet/model.json'
    const oModelPath = './public/model/Onet/model.json'
    const FacenetModel = await facenet.loadFacenetModel(modelPath)
    const mtcnnModel = await detectFace.loadModel(pModelPath, rModelPath, oModelPath)

    console.log("begin loding ")
    let trainVectors = await loadFiles(trainDatapath, FacenetModel, mtcnnModel[0], mtcnnModel[1], mtcnnModel[2])
    let predictVectors = await loadFiles(predictDatapath, FacenetModel, mtcnnModel[0], mtcnnModel[1], mtcnnModel[2])   
    let SVMTree = {}
    let index = trainVectors.slice(0,5).map((val, index)=>{
        return index
    })
    console.log(index)
    createTree(trainVectors.slice(0,5), predictVectors.slice(0,5), SVMTree, index)

    for(let i=0; i<10; i++){
        TreePredict(SVMTree, trainVectors[0][0])
    }
    for(let i=0; i<10; i++){
        TreePredict(SVMTree, trainVectors[1][0])
    }
    for(let i=0; i<10; i++){
        TreePredict(SVMTree, trainVectors[2][0])
    }
    for(let i=0; i<10; i++){
        TreePredict(SVMTree, trainVectors[3][0])
    }
    for(let i=0; i<10; i++){
        TreePredict(SVMTree, trainVectors[4][0])
    }
} 

SVMClassifier()