const tf = require("@tensorflow/tfjs-node");
const config = require("../configParameter/config.json")
const detectFace = require("./detectFace")
const fs = require("fs");
const alignFace = require("./alignFace")
const {Matrix, SingularValueDecomposition, covariance} = require('ml-matrix');

async function loadAgeModel(modelPath){
    const model = await tf.loadLayersModel('file://'+modelPath);
    return model
}

// async function lo(){
//     const [Pnet, Rnet, Onet] = await detectFace.loadModel(config.modelPath.pModelPath, config.modelPath.rModelPath, config.modelPath.oModelPath)
//     const model = await loadAgeModel(config.modelPath.genderAgeWideResnet)
//     //model.summary()
//     // const imagePath = "E:/tensorflow/age-gender-estimation-master/age_estimation/appa-real-release/train/000042.jpg"
//     // const imagePath = "C:/Users/1/Desktop/zzf3.jpg"
//     // const imagePath = "./public/images/TaylorSwift/TaylorSwift0001.jpg"
//     let img = fs.readFileSync(imagePath)
//     let imgTensor = tf.node.decodeImage(img)
//     if(imgTensor.shape[2] !== 3){
//         [imgTensor, useless] = tf.split(imgTensor, [3, 1], 2)
//         console.log(imgTensor)
//     }
//     let rectangles = detectFace.detectFace(imgTensor, config.mtcnnParam.threshold, Pnet, Rnet, Onet)
//     // let imgTensorFace = await alignFace.affineImage(imgTensor, rectangles)
//     let marginScale = 0.4
//     // if(imgTensorFace === "error"){
//     //let ageArr = []
//     //for(let i=0; i<10; i++){
//         //marginScale += i*0.1
//         let beginX = rectangles[0][1]-(rectangles[0][3]-rectangles[0][1])*marginScale<0 ? 0 : rectangles[0][1]-(rectangles[0][3]-rectangles[0][1])*marginScale
//         let beginY = rectangles[0][0]-(rectangles[0][2]-rectangles[0][0])*marginScale<0 ? 0 : rectangles[0][0]-(rectangles[0][2]-rectangles[0][0])*marginScale
//         let lengthX = (rectangles[0][3]-rectangles[0][1])*(1+marginScale) + beginX > imgTensor.shape[0] ? imgTensor.shape[0]-beginX : (rectangles[0][3]-rectangles[0][1])*(1+marginScale)
//         let lengthY = (rectangles[0][2]-rectangles[0][0])*(1+marginScale) + beginY > imgTensor.shape[1] ? imgTensor.shape[1]-beginY : (rectangles[0][2]-rectangles[0][0])*(1+marginScale)
//         let imgTensorFace = tf.slice(imgTensor, [beginX, beginY], [lengthX, lengthY])
//         // }
//         imgTensorFace = tf.image.resizeBilinear(imgTensorFace,[config.wideResnetParam.inputImageSize,config.wideResnetParam.inputImageSize])
//         let [r, g, b] = tf.split(imgTensorFace, [1,1,1], 2)
//         imgTensorFace = tf.concat([b,g,r], 2)
//         // console.log(imgTensorFace)
//         // imgTensorFace.print()
//         let input = imgTensorFace.reshape([1, config.wideResnetParam.inputImageSize, config.wideResnetParam.inputImageSize, 3])
//         let result = model.predict(input)
//         // console.log(result, typeof(result))
//         let predictGender = result[0].arraySync()[0]
//         if(predictGender[0]<0.5){
//             predictGender = "male"
//         }else{
//             predictGender = "female"
//         }
//         let ages = []
//         for(let i=0; i<101; i++){
//             ages.push([i])
//         }
//         let resMatrix = new Matrix(result[1].arraySync())
//         let agesMatrix = new Matrix(ages)
//         // console.log(resMatrix, agesMatrix)
//         let predictAge = resMatrix.mmul(agesMatrix).to1DArray()[0]
//         //ageArr.push(res)
//     //}
//     console.log("lo", predictGender, predictAge)
// }

// async function lo1(){
//     const [Pnet, Rnet, Onet] = await detectFace.loadModel(config.modelPath.pModelPath, config.modelPath.rModelPath, config.modelPath.oModelPath)
//     const model = await loadAgeModel(config.modelPath.ageResnet50)
//     //model.summary()
//     // const imagePath = "E:/tensorflow/age-gender-estimation-master/age_estimation/appa-real-release/train/000042.jpg"
//     // const imagePath = "C:/Users/1/Desktop/zzf3.jpg"
//     // const imagePath = "./public/images/TaylorSwift/TaylorSwift0001.jpg"
//     let img = fs.readFileSync(imagePath)
//     let imgTensor = tf.node.decodeImage(img)
//     if(imgTensor.shape[2] !== 3){
//         [imgTensor, useless] = tf.split(imgTensor, [3, 1], 2)
//         console.log(imgTensor)
//     }
//     let rectangles = detectFace.detectFace(imgTensor, config.mtcnnParam.threshold, Pnet, Rnet, Onet)
//     // let imgTensorFace = await alignFace.affineImage(imgTensor, rectangles)
//     let marginScale = 0.4
//     // if(imgTensorFace === "error"){
//     //let ageArr = []
//     //for(let i=0; i<10; i++){
//         //marginScale += i*0.1
//         let beginX = rectangles[0][1]-(rectangles[0][3]-rectangles[0][1])*marginScale<0 ? 0 : rectangles[0][1]-(rectangles[0][3]-rectangles[0][1])*marginScale
//         let beginY = rectangles[0][0]-(rectangles[0][2]-rectangles[0][0])*marginScale<0 ? 0 : rectangles[0][0]-(rectangles[0][2]-rectangles[0][0])*marginScale
//         let lengthX = (rectangles[0][3]-rectangles[0][1])*(1+marginScale) + beginX > imgTensor.shape[0] ? imgTensor.shape[0]-beginX : (rectangles[0][3]-rectangles[0][1])*(1+marginScale)
//         let lengthY = (rectangles[0][2]-rectangles[0][0])*(1+marginScale) + beginY > imgTensor.shape[1] ? imgTensor.shape[1]-beginY : (rectangles[0][2]-rectangles[0][0])*(1+marginScale)
//         let imgTensorFace = tf.slice(imgTensor, [beginX, beginY], [lengthX, lengthY])
//         // }
//         imgTensorFace = tf.image.resizeBilinear(imgTensorFace,[config.ageParam.inputImageSize,config.ageParam.inputImageSize])
//         let [r, g, b] = tf.split(imgTensorFace, [1,1,1], 2)
//         imgTensorFace = tf.concat([b,g,r], 2)
//         // console.log(imgTensorFace)
//         // imgTensorFace.print()
//         let input = imgTensorFace.reshape([1, config.ageParam.inputImageSize, config.ageParam.inputImageSize, 3])
//         let result = model.predict(input)
//         // console.log(result, typeof(result))
//         // let predictGender = result[0].arraySync()[0]
//         // if(predictGender[0]<0.5){
//         //     predictGender = "male"
//         // }else{
//         //     predictGender = "female"
//         // }
//         let ages = []
//         for(let i=0; i<101; i++){
//             ages.push([i])
//         }
//         let resMatrix = new Matrix(result.arraySync())
//         let agesMatrix = new Matrix(ages)
//         // console.log(resMatrix, agesMatrix)
//         let predictAge = resMatrix.mmul(agesMatrix).to1DArray()[0]
//         //ageArr.push(res)
//     //}
//     console.log("lo1", predictAge)
// }

function genderAgePredict(imagePath, Pnet, Rnet, Onet, ageModel, genderModel, marginScale){
    // const [Pnet, Rnet, Onet] = await detectFace.loadModel(config.modelPath.pModelPath, config.modelPath.rModelPath, config.modelPath.oModelPath)
    // const ageModel = await loadAgeModel(config.modelPath.ageResnet50)
    // const genderModel = await loadAgeModel(config.modelPath.genderAgeWideResnet)

    //model.summary()
    // const imagePath = "E:/tensorflow/age-gender-estimation-master/age_estimation/appa-real-release/train/000042.jpg"
    // const imagePath = "C:/Users/1/Desktop/zzf3.jpg"
    // const imagePath = "./public/images/TaylorSwift/TaylorSwift0001.jpg"
    let img = fs.readFileSync(imagePath)
    let imgTensor = tf.node.decodeImage(img)
    if(imgTensor.shape[2] !== 3){
        [imgTensor, useless] = tf.split(imgTensor, [3, 1], 2)
        console.log(imgTensor)
    }
    let rectangles = detectFace.detectFace(imgTensor, config.mtcnnParam.threshold, Pnet, Rnet, Onet)
    // let imgTensorFace = await alignFace.affineImage(imgTensor, rectangles)
    let beginX = rectangles[0][1]-(rectangles[0][3]-rectangles[0][1])*marginScale<0 ? 0 : rectangles[0][1]-(rectangles[0][3]-rectangles[0][1])*marginScale
    let beginY = rectangles[0][0]-(rectangles[0][2]-rectangles[0][0])*marginScale<0 ? 0 : rectangles[0][0]-(rectangles[0][2]-rectangles[0][0])*marginScale
    let lengthX = (rectangles[0][3]-rectangles[0][1])*(1+marginScale) + beginX > imgTensor.shape[0] ? imgTensor.shape[0]-beginX : (rectangles[0][3]-rectangles[0][1])*(1+marginScale)
    let lengthY = (rectangles[0][2]-rectangles[0][0])*(1+marginScale) + beginY > imgTensor.shape[1] ? imgTensor.shape[1]-beginY : (rectangles[0][2]-rectangles[0][0])*(1+marginScale)
    let imgTensorFace = tf.slice(imgTensor, [beginX, beginY], [lengthX, lengthY])
    let [r, g, b] = tf.split(imgTensorFace, [1,1,1], 2)
    imgTensorFace = tf.concat([b,g,r], 2)
    let genderImgTensorFace = tf.image.resizeBilinear(imgTensorFace,[config.wideResnetParam.inputImageSize,config.wideResnetParam.inputImageSize])
    let genderInput = genderImgTensorFace.reshape([1, config.wideResnetParam.inputImageSize, config.wideResnetParam.inputImageSize, 3])
    let gResult = genderModel.predict(genderInput)
    let predictGender = gResult[0].arraySync()[0]
    if(predictGender[0]<0.5){
        predictGender = "male"
    }else{
        predictGender = "female"
    }

    let ageImgTensorFace = tf.image.resizeBilinear(imgTensorFace,[config.ageParam.inputImageSize,config.ageParam.inputImageSize])
    let ageInput = ageImgTensorFace.reshape([1, config.ageParam.inputImageSize, config.ageParam.inputImageSize, 3])
    let aResult = ageModel.predict(ageInput).arraySync()

    // //利用矩阵计算结果 
    // let ages = []
    // for(let i=0; i<101; i++){
    //     ages.push([i])
    // }
    // let resMatrix = new Matrix(aResult)
    // let agesMatrix = new Matrix(ages)
    // // console.log(resMatrix, agesMatrix)
    // let predictAge = resMatrix.mmul(agesMatrix).to1DArray()[0]
    // //ageArr.push(res)
    let predictAge = 0
    for(let i=0; i<101; i++){
        predictAge += aResult[0][i] * i
    }
    return [predictGender, predictAge]
}

async function loadModel(){
    const [Pnet, Rnet, Onet] = await detectFace.loadModel(config.modelPath.pModelPath, config.modelPath.rModelPath, config.modelPath.oModelPath)
    const ageModel = await loadAgeModel(config.modelPath.ageResnet50)
    const genderModel = await loadAgeModel(config.modelPath.genderAgeWideResnet)
    return [Pnet, Rnet, Onet, ageModel, genderModel]
}

async function main(){
    const [Pnet, Rnet, Onet, ageModel, genderModel] = await loadModel()
    const imagePath = "E:/tensorflow/age-gender-estimation-master/age_estimation/appa-real-release/train/000073.jpg"
    let marginScale = 0.1
    for(let i=0; i<10; i++){
        const [gender, age] = genderAgePredict(imagePath, Pnet, Rnet, Onet, ageModel, genderModel, marginScale + i*0.1)
        console.log(gender, age)
    }
}

main()