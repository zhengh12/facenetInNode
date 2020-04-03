const {Matrix, SingularValueDecomposition, covariance} = require('ml-matrix');
const tf = require("@tensorflow/tfjs-node");
const ia = require('image-augment')(tf) //进行仿射和图像增强的包
const h = require('hasard')
const fs = require('fs')
const {PNG} = require('pngjs')
const config = require("../configParameter/config.json")

// const coord5point = [[30.2946, 51.6963],
// [65.5318, 51.6963],
// [48.0252, 71.7366],
// [33.5493, 92.3655],
// [62.7299, 92.3655]]

// const coord5point1 = [[0, 0], [2, 0], [1, 1], [0, 2], [2, 2]]
// const face_landmarks1 = [[2, 0], [4, 2], [2, 2], [0, 2], [2, 4]]
// const face_landmarks2 = [[3, 0], [4, 1], [2, 2], [0, 3], [1, 4]]
// const face_landmarks1 = [[0, 0], [4, 0], [2, 2], [0, 4], [4, 4]]

// const face_landmarks = [[70, 111],
// [110, 113],
// [100, 134],
// [68, 152],
// [111, 153]]

//通过两组地标点获得仿射矩阵并获得仿射所需要的角度，缩放，平移数值
//input parameters: 
//  Number Array[5][2] points1 原始的五个地标点
//  Number Array[5][2] points2 目标的五个地标点
//output:
//  Number Array[3] [angle, scale, distance] 角度，缩放，平移值
function getAffineFactor(points1, points2){
    let mean1 = [...''.padEnd(2)].map((val,index) => {
        return points1.reduce((a, b)=>{
            return a + b[index]
        }, 0) / points1.length
    })
    let mean2 = [...''.padEnd(2)].map((val,index) => {
        return points2.reduce((a, b)=>{
            return a + b[index]
        }, 0) / points2.length
    })
    let std1 = Math.sqrt(points1.reduce((a, b)=>{
        return a + b.reduce((a, b, index)=>{
            return a + Math.pow(b - mean1[index], 2)
        }, 0)
    }, 0) / (points1.length * mean1.length))
    let std2 = Math.sqrt(points2.reduce((a, b)=>{
        return a + b.reduce((a, b, index)=>{
            return a + Math.pow(b - mean2[index], 2)
        }, 0)
    }, 0) / (points2.length * mean2.length))
    let p1 = points1.map(val=>{
        return val.map((val, index)=>{
            return (val - mean1[index]) / std1
        })
    })
    let p2 = points2.map(val=>{
        return val.map((val, index)=>{
            return (val - mean2[index]) / std2
        })
    })
    let Matrix1 = new Matrix(p1)
    let Matrix2 = new Matrix(p2)
    let mMatrix1 = new Matrix([mean1])
    let mMatrix2 = new Matrix([mean2])
    let CMatrix = new Matrix([[1, 0], [0, -1]]) //由于ml-matrix的奇异值分解结果和numpy.linalg.svd的结果符号不同所以需要乘以该矩阵来调整符号
    let SingularValueMatrix = new SingularValueDecomposition(Matrix1.transpose().mmul(Matrix2)) 
    let R = SingularValueMatrix.leftSingularVectors.mmul(CMatrix).mmul(SingularValueMatrix.rightSingularVectors.mmul(CMatrix)).transpose()
    let angleAndScale = R.mul(std2/std1).to2DArray()
    let distance = mMatrix2.transpose().sub(R.mmul(mMatrix1.transpose())).to1DArray()
    let scale = std2/std1
    let angle = Math.asin(angleAndScale[0][1] / scale) / Math.PI * 180
    // let scale = angleAndScale[0][0] / Math.sqrt(1 - Math.pow(angle, 2))
    // angleAndScale[0].push(distance[0])
    // angleAndScale[1].push(distance[1])
    // let affineMatrix = new Matrix(angleAndScale)
    // console.log(affineMatrix)
    // let point = []
    // for(let i=0; i<4; i++){
    //     for(let j=0; j<4; j++){
    //         point.push([i, j, 1])
    //     }
    // }
    // for(val of points1){
    //     val.push(1)
    // }
    // console.log(points1)
    // console.log(new Matrix(point).transpose())
    // let res = affineMatrix.mmul(new Matrix(point).transpose()).transpose().to2DArray()
    // res = res.map(val=>{
    //     return val.map(val=>{
    //         return Math.floor(val)
    //     })
    // })
    // console.log(res)
    return [angle, scale, distance]
}

//将仿射结果输出成png图像文件
//input parameters: 
//  Number Array[n][15] rectangles mtcnn得到的人脸矩形区域和地标点数组
//  tf.Tensor images 原始图像
//output:
//  一张png图像
const tensorToFile = function (filename, images) {
	return new Promise((resolve, reject) => {
		const png = new PNG({
			width: images.shape[2],
			height: images.shape[1]
		});
		png.data = images.dataSync();
		png
			.pack()
			.pipe(fs.createWriteStream(filename))
			.on('error', reject)
			.on('close', resolve);
	});
};

// [angle, scale, distance] = getAffineFactor(face_landmarks1, coord5point1)
// console.log("angle: ", angle, "scale: ", scale, "distance: ", distance)

//通过5个人脸地标和预设人脸位置进行对人脸图像进行仿射变换
//warning: 虽然可以得到正确的变换矩阵, 但目前js中没有发现将变换矩阵应用于仿射的方法。所以根据变换矩阵求出三个仿射过程的重要数值，角度，缩放，平移。
//目前仅应用了角度，即将人脸旋转对齐到正确位置上，其他两个参数还未应用。所以函数中所截取的正方形仅支持旋转不溢出。
//input parameters: 
//  tf.Tensor images 原始图像
//  Number Array[n][15] rectangles mtcnn得到的人脸矩形区域和地标点数组
//output:
//  tf.Tensor t1 人脸仿射对齐后的图像
async function affineImage(images, rectangles){
    let rectangle = rectangles[0]
    const centerx = (rectangle[2] - rectangle[0]) / 2 + rectangle[0]
    const centery = (rectangle[3] - rectangle[1]) / 2 + rectangle[1]
    const squareLenght = Math.sqrt(Math.pow((rectangle[2] - rectangle[0]),2) + Math.pow((rectangle[3] - rectangle[1]),2))
    const leftTop = [centerx - squareLenght / 2, centery - squareLenght / 2]
    if(leftTop[0]>0 && leftTop[1]>0 && leftTop[1] + squareLenght < images.shape[0] && leftTop[0] + squareLenght < images.shape[1]){
        let sub = []
        for(let j=0; j<images.shape[1]; j++){
            sub.push([225])  
        }
        let aImageArr = []
        for(let j=0; j<images.shape[0]; j++){
            aImageArr.push([...sub])  
        }
        aImageArr = tf.tensor(aImageArr,[images.shape[0], images.shape[1], 1],"int32")
        images = tf.concat([images, aImageArr], 2) //由于image-augment需要输入的tensor在在第四维有RGB和透明度 所以这里将255拼接到第四维上
        images = images.slice([leftTop[1], leftTop[0]], [squareLenght, squareLenght])
        images = images.as4D(1, images.shape[0], images.shape[1], 4)
        // images.print()//000275.jpg眼镜
        //默认将鼻子当做预设人脸的中心点
        let offsetEyeX = (centerx - rectangle[0]) * config.faceAlignParam.eyeFactorx //eyeFactorx = 0.5 预设的眼睛位置，眼睛到鼻子与眼睛到人脸矩形边界距离的x轴比值,自定义。
        let offsetEyeY = (centery - rectangle[1]) * config.faceAlignParam.eyeFactory //eyeFactory = 0.5 预设的眼睛位置，眼睛到鼻子与眼睛到人脸矩形边界距离的y轴比值,自定义。
        let offsetMouX = (centerx - rectangle[0]) * config.faceAlignParam.mouFactorx //mouFactorx = 0.5 预设的嘴巴位置，嘴巴端点到鼻子与嘴巴端点到人脸矩形边界距离的x轴比值,自定义。
        let offsetMouY = (centery - rectangle[1]) * config.faceAlignParam.mouFactory //mouFactory = 0.5 预设的嘴巴位置，嘴巴端点到鼻子与嘴巴端点到人脸矩形边界距离的y轴比值,自定义。
        const [angle, scale, distance] = getAffineFactor(
            [[rectangle[5], rectangle[6]],[rectangle[7], rectangle[8]],[rectangle[9], rectangle[10]],[rectangle[11], rectangle[12]],[rectangle[13], rectangle[14]]],
            [[centerx-offsetEyeX, centery-offsetEyeY],[centerx+offsetEyeX, centery-offsetEyeY],[centerx, centery],[centerx-offsetMouX, centery+offsetMouY],[centerx+offsetMouX, centery+offsetMouY]])
        //console.log("angle: ", angle, "scale: ", scale, "distance: ", distance)
        if(Math.abs(angle) > config.faceAlignParam.smallestAngle){
            const basicAugmentation = ia.sequential([
                ia.affine({translatePercent: [0,0],rotate: -angle, scale:1}),
                ia.blur(1)
            ]);
            affineImages = await basicAugmentation.read({images:images})
            affineImages = affineImages.images
            affineImages = affineImages.slice([0, rectangle[1]-leftTop[1], rectangle[0]-leftTop[0]],[1, rectangle[3] - rectangle[1], rectangle[2] - rectangle[0]])
            await tensorToFile('./public/images/zheng1.png', affineImages) //输出成图像可视化对齐结果
            let [t1, t2] = tf.split(affineImages, [3,1], 3) //分成3,1是因为前三个为透明度
            t1 = t1.reshape([t1.shape[1], t1.shape[2], 3])
            return t1
        }else{
            //console.log('仿射角度小于' + config.faceAlignParam.smallestAngle + '，使用仿射对结果并无太大改善！')
            return "error"
        }
    }else{
        //console.log('人脸区域靠近图像边缘以至于无法转换成正方形！')
        return "error"
    }
}

exports.affineImage = affineImage
