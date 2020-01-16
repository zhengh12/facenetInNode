const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const images = require("images");
const detectFace = require("./detectFace")
const alignFace = require("./alignFace")

// "./public/images/test/微信图片2.jpg"
const image1Path = "./public/images/RandomForestTrainData/TaylorSwift/TaylorSwift0003.jpg"
const image2Path = "./public/images/tywai.jpeg"
const modelPath = "./public/model/Facenet1/model.json"

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

//通过facenet网络获得两个输入图像的特征脸向量。
//input parameters: 
//  modelPath 网络拓扑结构json文件地址
//  image1Path 图像1的文件地址
//  image2Path 图像2的文件地址
//output:
//  一个长度为2的张量数组
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

    // //根据mtcnn人脸检测到的矩形从原始图像中截取人脸数据，若多人脸也只取第一个矩形也就是最像人脸的区域
    // imgTensor = imgTensor.arraySync()
    // let imgTensorFace = []
    // imgTensor.map((val,index)=>{
    //     if(index>rectangles[0][1]&&index<=rectangles[0][3]){
    //         imgTensorFace.push(val.slice(rectangles[0][0],rectangles[0][2]))
    //     }
    // })
    // // imgTensorFace = imgTensorFace.map(val=>{
    // //     return val.map(val=>{
    // //         let array = val.map(val=>{
    // //             return val
    // //         })
    // //         array.splice(2,1,...array.splice(0, 1 , array[2]));
    // //         return array
    // //     })
    // // })

    // imgTensor1 = imgTensor1.arraySync()
    // let imgTensor1Face = []
    // imgTensor1.map((val,index)=>{
    //     if(index>rectangles1[0][1]&&index<=rectangles1[0][3]){
    //         imgTensor1Face.push(val.slice(rectangles1[0][0],rectangles1[0][2]))
    //     }
    // })
    // // imgTensor1Face = imgTensor1Face.map(val=>{
    // //     return val.map(val=>{
    // //         let array = val.map(val=>{
    // //             return val
    // //         })
    // //         array.splice(2,1,...array.splice(0, 1 , array[2]));
    // //         return array
    // //     })
    // // })

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
    return [out,out1]
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

//转化单张图片的函数
function faceVector(model, imagePath, Pnet, Rnet, Onet){
    let img = fs.readFileSync(imagePath)
    let imgTensor = tf.node.decodeImage(img)
    let threshold = [0.6,0.6,0.7]
    let rectangles = detectFace.detectFace(imgTensor, threshold, Pnet, Rnet, Onet)
    imgTensor = imgTensor.arraySync()
    let imgTensorFace = []
    imgTensor.map((val,index)=>{
        if(index>rectangles[0][1]&&index<=rectangles[0][3]){
            imgTensorFace.push(val.slice(rectangles[0][0],rectangles[0][2]))
        }
    })
    imgTensorFace = tf.image.resizeBilinear(tf.tensor(imgTensorFace,[rectangles[0][3]-rectangles[0][1],rectangles[0][2]-rectangles[0][0],3]),[160,160])
    imgTensorFace = prewhiten(imgTensorFace)
    let input = imgTensorFace.reshape([1,160,160,3])
    let out = model.predict(input)
    out = l2_normalize(out)
    // out.print()
    return out
}

async function faceAlignVector(model, imagePath, Pnet, Rnet, Onet){
    let img = fs.readFileSync(imagePath)
    let imgTensor = tf.node.decodeImage(img)
    let threshold = [0.6,0.6,0.7]
    let rectangles = detectFace.detectFace(imgTensor, threshold, Pnet, Rnet, Onet)
    let imgTensorFace = await alignFace.affineImage(imgTensor, rectangles)
    imgTensorFace = tf.image.resizeBilinear(imgTensorFace, [160,160])
    imgTensorFace = prewhiten(imgTensorFace)
    let input = imgTensorFace.reshape([1,160,160,3])
    let out = model.predict(input)
    out = l2_normalize(out)
    // out.print()
    return out
}

async function test(){
    const pModelPath = './public/model/Pnet/model.json'
    const rModelPath = './public/model/Rnet/model.json'
    const oModelPath = './public/model/Onet/model.json'
    const mtcnnModel = await detectFace.loadModel(pModelPath, rModelPath, oModelPath)
    const model = await loadFacenetModel(modelPath)
    let vector1 = await faceVector(model, image1Path, mtcnnModel[0], mtcnnModel[1], mtcnnModel[2])
    let vector2 = await faceVector(model, image2Path, mtcnnModel[0], mtcnnModel[1], mtcnnModel[2])
    let dist1 = tf.sqrt(tf.sum(tf.squaredDifference(vector1,vector2)))
    dist1.print()
    let AlignedVector1 = await faceAlignVector(model, image1Path, mtcnnModel[0], mtcnnModel[1], mtcnnModel[2])
    let AlignedVector2 = await faceAlignVector(model, image2Path, mtcnnModel[0], mtcnnModel[1], mtcnnModel[2])
    let dist2 = tf.sqrt(tf.sum(tf.squaredDifference(AlignedVector1,AlignedVector2)))
    dist2.print()
    
}
test()

exports.faceVector = faceVector
exports.loadFacenetModel = loadFacenetModel
exports.prewhiten = prewhiten
exports.l2_normalize = l2_normalize
