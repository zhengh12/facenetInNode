
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const images = require("images");
const detectFace = require("./detectFace")
// const inceptionResNetV2 = require("./inceptionResNetV2")
const config = require("../configParameter/config.json")
const alignFace = require("./alignFace")
//const dataGagenerator = require("./dataGagenerator")
const filePath = "./public/images/tywai.jpeg"

//该函数以图像输出来观察mtcnn人脸检测结果
//output:
//  带标注的图像
async function testMtcnn(){
    let img = fs.readFileSync(filePath)
    let imgarr = tf.node.decodeImage(img)
    //let imgTensors = imgTensor.reshape([1,224,224,3])
    let [Pnet, Rnet, Onet] = await detectFace.loadModel(config.modelPath.pModelPath, config.modelPath.rModelPath, config.modelPath.oModelPath)
    let rectangles = detectFace.detectFace(imgarr, config.mtcnnParam.threshold, Pnet, Rnet, Onet)
    console.log('finalrectangles:', rectangles)
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
    image.save("./public/images/mtcnnResult.jpg")

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

testMtcnn()
exports.testMtcnn = testMtcnn

