
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const images = require("images");
const detectFace = require("./detectFace")
// const inceptionResNetV2 = require("./inceptionResNetV2")
const config = require("../configParameter/config.json")
const alignFace = require("./alignFace")
// const pModelPath = './public/model/Pnet/model.json'
// const rModelPath = './public/model/Rnet/model.json'
// const oModelPath = './public/model/Onet/model.json'
//const dataGagenerator = require("./dataGagenerator")
const filePath = "./public/images/tywai.jpeg"

async function loadLayersModel(){
    // images("./public/images/goldfinch, Carduelis carduelis/hjhj.jpg")
    // .resize(224,224)
    // .save("./public/images/goldfinch, Carduelis carduelis/hjhj.jpeg")
    //let img = readImg("C:/Users/1/Desktop/tensorflowjs/tfjsNode/public/images/TaylorSwift/TaylorSwift1.png")
    //let img2 = fs.readFileSync("./1.png")
    let img = fs.readFileSync(filePath)
    //let img = fs.readFileSync("C:/Users/1/Desktop/tensorflowjs/tfjsNode/tfjsInNode/public/images/TaylorSwift/TaylorSwift1.png")
    //tfjs/mobilenet_v1_0.25_224/imagenet_class_names.json
    //let img = new Image("C:/Users/1/Desktop/tensorflowjs/tfjsNode/public/images/TaylorSwift/TaylorSwift1.png")
    //let u8 = new Uint8Array(img)
    //console.log(imgg)
    let imgarr = tf.node.decodeImage(img)
    //let imgTensors = imgTensor.reshape([1,224,224,3])
    let mtcnnModel = await detectFace.loadModel(config.modelPath.pModelPath, config.modelPath.rModelPath, config.modelPath.oModelPath)
    let rectangles = detectFace.detectFace(imgarr,config.mtcnnParam.threshold,mtcnnModel[0], mtcnnModel[1], mtcnnModel[2])
    console.log('finalrectangles:',rectangles)
    await alignFace.affineImage(imgarr, rectangles)
    let image = images(filePath)
    rectangles.map(val=>{
        let x1 = val[0]
        let y1 = val[1]
        let x2 = val[2]
        let y2 = val[3]
        let imgborder = 5
        console.log(x1,y1,x2-x1,y2-y1)
        image.draw(images(x2-x1, imgborder).fill(127, 255, 170, 0.7),x1,y1)
        .draw(images(imgborder, y2-y1).fill(127, 255, 170, 0.7),x1,y1)
        .draw(images(x2-x1, imgborder).fill(127, 255, 170, 0.7),x1,y2-imgborder)
        .draw(images(imgborder, y2-y1).fill(127, 255, 170, 0.7),x2-imgborder,y1)
    })
    image.save("./public/Tyler.jpeg")

    // const model = await tf.loadLayersModel('file://./public/model/model.json');

    // let model = inceptionResNetV2.create_inception_resnet_v2()
    //model.fitDataset(dataGagenerator.ds)
    // model.summary()
    
    // let arr = await model.predict(imgTensors).array()
    // let max = 0
    // let maxflag = 0
    // for(let i=0; i<arr[0].length; i++){
    //     if (max < arr[0][i]){
    //         max = arr[0][i]
    //         maxflag = i
    //     }
    // }
    // console.log(maxflag)
    //model.loadWeights('file://C:/Users/1/Desktop/tensorflowjs/tfjsNode/tfjsInNode/public/model/ssd_mobilenetv1_model-weights_manifest.json')
    // return model
}

let model1 = loadLayersModel()
exports.loadLayersModel = loadLayersModel

