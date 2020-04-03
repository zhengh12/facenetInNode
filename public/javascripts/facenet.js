const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const images = require("images");
const detectFace = require("./detectFace")
const alignFace = require("./alignFace")
const config = require("../configParameter/config.json") //导入配置参数文件
const image1Path = "./public/images/TaylorSwift/TaylorSwift0001.jpg"
const image2Path = "E:/tensorflow/age-gender-estimation-master/age_estimation/appa-real-release/train/000001.jpg"
//E:/tensorflow/Keras_TP-GAN-master/images/x2.png

//导入facenet网络模型
async function loadFacenetModel(modelPath){
    //自定义layer类来对数据进行自定义处理并给其定义一个静态className变量
    class Lambda017 extends tf.layers.Layer {
        constructor() {
            super({});
        }
        // In this case, the output is a scalar.
        computeOutputShape(input) { return input; }
        // call() is where we do the computation.
        call(input,kwargs) {
            return tf.mul(input[0],tf.scalar(0.17))
        } 
        // Every layer needs a unique name.
        getClassName() { return 'Lambda'; }  
    }
    Lambda017.className = 'Lambda017'

    class Lambda01 extends tf.layers.Layer {
        constructor() {
            super({});
        }        
        // In this case, the output is a scalar.
        computeOutputShape(input) { return input; }
        // call() is where we do the computation.
        call(input,kwargs) {
            return tf.mul(input[0],tf.scalar(0.1))
        }      
        // Every layer needs a unique name.
        getClassName() { return 'Lambda'; }      
    }
    Lambda01.className = 'Lambda01'

    class Lambda02 extends tf.layers.Layer {       
        constructor() {
            super({});
        }  
        // In this case, the output is a scalar.
        computeOutputShape(input) { return input; }    
        // call() is where we do the computation.
        call(input,kwargs) {
            return tf.mul(input[0],tf.scalar(0.2))
        } 
        // Every layer needs a unique name.
        getClassName() { return 'Lambda'; }     
    }
    Lambda02.className = 'Lambda02'

    class Lambda1 extends tf.layers.Layer {        
        constructor() {
            super({});
        }     
        // In this case, the output is a scalar.
        computeOutputShape(input) { return input; }    
        // call() is where we do the computation.
        call(input,kwargs) {
            return tf.mul(input[0],tf.scalar(1.0))
        }       
        // Every layer needs a unique name.
        getClassName() { return 'Lambda'; }
    }
    Lambda1.className = 'Lambda1'
    
    //注册自定义层
    tf.serialization.registerClass(Lambda017);
    tf.serialization.registerClass(Lambda01);
    tf.serialization.registerClass(Lambda02);
    tf.serialization.registerClass(Lambda1);

    const model = await tf.loadLayersModel('file://'+modelPath);
    return model
}

//比对两个输入图像的人脸距离。
//input parameters: 
//  String modelPath 网络拓扑结构json文件地址
//  String image1Path 图像1的文件地址
//  String image2Path 图像2的文件地址
//  tf.LayersModel Pnet mtcnn快速扫描人脸区域的Pnet
//  tf.LayersModel Rnet mtcnn精确判断人脸区域的Rnet
//  tf.LayersModel Pnet mtcnn定位人脸地标位置的Onet
//output:
//  tf.Tensor dis 一个长度为1的张量
async function EigenfaceVector(modelPath, image1Path, image2Path, Pnet, Rnet, Onet){
    
    //导入模型和图像数据
    let model = await loadFacenetModel(modelPath)
    let img = fs.readFileSync(image1Path)
    let img1 = fs.readFileSync(image2Path)
    let imgTensor = tf.node.decodeImage(img)
    let imgTensor1 = tf.node.decodeImage(img1)
    let threshold = [0.6,0.7,0.7]

    //利用deteFace函数中的mtcnn网络获取图像的人脸矩形区域
    let rectangles = await detectFace.detectFace(imgTensor, threshold, Pnet, Rnet, Onet)
    let rectangles1 = await detectFace.detectFace(imgTensor1,threshold, Pnet, Rnet, Onet)
    console.log('rectanglesout:',rectangles)
    console.log('rectangles1out:',rectangles1)

    //根据mtcnn人脸检测到的矩形从原始图像中截取人脸数据，若多人脸也只取第一个矩形也就是最像人脸的区域
    //使用矩阵仿射方法进行人脸对齐
    let imgTensorFace = await alignFace.affineImage(imgTensor, rectangles)
    let imgTensor1Face = await alignFace.affineImage(imgTensor1, rectangles1)

    //将输入标准化使其符合标准正态分布
    imgTensorFace = prewhiten(imgTensorFace)
    let input = imgTensorFace.reshape([1,160,160,3])

    //将输入标准化使其符合标准正态分布
    imgTensor1Face = prewhiten(imgTensor1Face)
    let input1 = imgTensor1Face.reshape([1,160,160,3])

    //通过fecenet网络预测并将结果正则化
    let out = model.predict(input)
    let out1 = model.predict(input1)
    //将结果l2正则化
    out = l2_normalize(out)
    out1 = l2_normalize(out1)
    console.log("l2_normalize:")
    out.print()
    out1.print()
    //计算欧式距离并输出
    let dist = tf.sqrt(tf.sum(tf.squaredDifference(out,out1)))
    dist.print()
}

//该函数将数据l2正则化,可使用tf.regularizers.l2
function l2_normalize(x, axis=-1, epsilon=1e-10){
    let output = tf.div(x, tf.sqrt(tf.maximum(tf.sum(tf.square(x), axis, true), tf.scalar(epsilon))))
    return output
}

//该函数将数据标准化,取均值和标准差的时候都是计算全局
function prewhiten(x) {
    let axis = [0, 1, 2]
    let size = x.size
    let mean = tf.mean(x, axis)
    mean = mean.arraySync()
    x = x.arraySync()
    let sum = 0
    for(let i=0; i<x.length; i++){
        for(let j=0; j<x[0].length; j++){
            for(let k=0; k<x[0][0].length; k++){
                sum = sum + Math.pow(x[i][j][k] - mean,2)
            }
        }
    }
    let std = Math.sqrt(sum/size)
    let y = x.map(val=>{
        return val.map(val=>{
            return val.map(val=>{
                return (val-mean)/std
            })

        })
    })
    y = tf.tensor(y)
    return y
}

//获得单张输入图像的特征脸向量。
//input parameters: 
//  String modelPath 网络拓扑结构json文件地址
//  String imagePath 图片的所在路径
//  tf.LayersModel Pnet mtcnn快速扫描人脸区域的Pnet
//  tf.LayersModel Rnet mtcnn精确判断人脸区域的Rnet
//  tf.LayersModel Pnet mtcnn定位人脸地标位置的Onet
//output:
//  tf.Tensor out 一个长度为128的张量
function faceVector(model, imagePath, Pnet, Rnet, Onet){
    let img = fs.readFileSync(imagePath)
    let imgTensor = tf.node.decodeImage(img)
    if(imgTensor.shape[2] !== 3){
        [imgTensor, useless] = tf.split(imgTensor, [3, 1], 2)
        console.log(imgTensor)
    }
    let rectangles = detectFace.detectFace(imgTensor, config.mtcnnParam.threshold, Pnet, Rnet, Onet)
    imgTensor = imgTensor.arraySync()
    let imgTensorFace = []
    imgTensor.map((val,index)=>{
        if(index>rectangles[0][1]&&index<=rectangles[0][3]){
            imgTensorFace.push(val.slice(rectangles[0][0],rectangles[0][2]))
        }
    })
    imgTensorFace = tf.image.resizeBilinear(tf.tensor(imgTensorFace,[rectangles[0][3]-rectangles[0][1],rectangles[0][2]-rectangles[0][0],3]),[config.facenetParam.inputImageSize,config.facenetParam.inputImageSize])
    imgTensorFace = prewhiten(imgTensorFace)
    let input = imgTensorFace.reshape([1, config.facenetParam.inputImageSize, config.facenetParam.inputImageSize, 3])
    let out = model.predict(input)
    out = l2_normalize(out)
    // out.print()
    return out
}

//加上人脸仿射获得单张输入图像的特征脸向量。
//input parameters: 
//  String modelPath 网络拓扑结构json文件地址
//  String imagePath 图片的所在路径
//  tf.LayersModel Pnet mtcnn快速扫描人脸区域的Pnet
//  tf.LayersModel Rnet mtcnn精确判断人脸区域的Rnet
//  tf.LayersModel Pnet mtcnn定位人脸地标位置的Onet
//output:
//  tf.Tensor out 一个长度为128的张量
async function faceAlignVector(model, imagePath, Pnet, Rnet, Onet){
    let img = fs.readFileSync(imagePath)
    let imgTensor = tf.node.decodeImage(img)
    if(imgTensor.shape[2] !== 3){
        [imgTensor, useless] = tf.split(imgTensor, [3, 1], 2)
        console.log(imgTensor)
    }
    let rectangles = detectFace.detectFace(imgTensor, config.mtcnnParam.threshold, Pnet, Rnet, Onet)
    let imgTensorFace = await alignFace.affineImage(imgTensor, rectangles)
    if(imgTensorFace !== "error"){
        imgTensorFace = tf.image.resizeBilinear(imgTensorFace, [config.facenetParam.inputImageSize,config.facenetParam.inputImageSize])
        imgTensorFace = prewhiten(imgTensorFace)
        let input = imgTensorFace.reshape([1, config.facenetParam.inputImageSize, config.facenetParam.inputImageSize, 3])
        let out = model.predict(input)
        out = l2_normalize(out)
        // out.print()
        return out
    }else{
        //console.log("放弃使用人脸对齐函数！")
        imgTensorFace = tf.slice(imgTensor, [rectangles[0][1], rectangles[0][0]], [rectangles[0][3]-rectangles[0][1], rectangles[0][2]-rectangles[0][0]])
        imgTensorFace = tf.image.resizeBilinear(imgTensorFace,[config.facenetParam.inputImageSize,config.facenetParam.inputImageSize])
        imgTensorFace = prewhiten(imgTensorFace)
        let input = imgTensorFace.reshape([1, config.facenetParam.inputImageSize, config.facenetParam.inputImageSize, 3])
        let out = model.predict(input)
        out = l2_normalize(out)
        // out.print()
        return out
    }
}

//测试两张人脸图片之间的欧式距离
async function test(){
    const [Pnet, Rnet, Onet] = await detectFace.loadModel(config.modelPath.pModelPath, config.modelPath.rModelPath, config.modelPath.oModelPath)
    const model = await loadFacenetModel(config.modelPath.facenetModel)
    // console.time('run 1 time')
    let vector1 = await faceVector(model, image1Path, Pnet, Rnet, Onet)
    let vector2 = await faceVector(model, image2Path, Pnet, Rnet, Onet)
    let dist1 = tf.sqrt(tf.sum(tf.squaredDifference(vector1,vector2)))
    dist1.print()
    // console.timeEnd('run 1 time')
    // console.time('run 2 time')
    let AlignedVector1 = await faceAlignVector(model, image1Path, Pnet, Rnet, Onet)
    let AlignedVector2 = await faceAlignVector(model, image2Path, Pnet, Rnet, Onet)
    let dist2 = tf.sqrt(tf.sum(tf.squaredDifference(AlignedVector1,AlignedVector2)))
    dist2.print()
    // console.timeEnd('run 2 time')
}
test()

exports.faceVector = faceVector
exports.faceAlignVector = faceAlignVector
exports.loadFacenetModel = loadFacenetModel
exports.prewhiten = prewhiten
exports.l2_normalize = l2_normalize
