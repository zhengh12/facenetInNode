const {Matrix, SingularValueDecomposition, covariance} = require('ml-matrix');
const tf = require("@tensorflow/tfjs-node");
const ia = require('image-augment')(tf)
const h = require('hasard')
const fs = require('fs')
const {PNG} = require('pngjs')

const coord5point = [[30.2946, 51.6963],
[65.5318, 51.6963],
[48.0252, 71.7366],
[33.5493, 92.3655],
[62.7299, 92.3655]]

const coord5point1 = [[0, 0], [2, 0], [1, 1], [0, 2], [2, 2]]
const face_landmarks1 = [[2, 0], [4, 2], [2, 2], [0, 2], [2, 4]]
const face_landmarks2 = [[3, 0], [4, 1], [2, 2], [0, 3], [1, 4]]
// const face_landmarks1 = [[0, 0], [4, 0], [2, 2], [0, 4], [4, 4]]

const face_landmarks = [[70, 111],
[110, 113],
[100, 134],
[68, 152],
[111, 153]]


function getAffineFactor(points1, points2){
    console.log("p1:", points1, points2)
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
    console.log(angleAndScale, distance)
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

// const fileToTensor = function (filename) {
// 	return new Promise((resolve, reject) => {
// 		const inputPng = new PNG();
// 		fs.createReadStream(filename)
// 			.pipe(inputPng)
// 			.on('parsed', () => {
//                 const images = tf.tensor4d(inputPng.data, [1, inputPng.height, inputPng.width, 4]);
// 				resolve({images});
// 			})
// 			.on('error', reject);
// 	});
// };

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

async function affineImage(images, rectangles){
    let rectangle = rectangles[0]
    const centerx = (rectangle[2] - rectangle[0]) / 2 + rectangle[0]
    const centery = (rectangle[3] - rectangle[1]) / 2 + rectangle[1]
    const squareLenght = Math.sqrt(Math.pow((rectangle[2] - rectangle[0]),2) + Math.pow((rectangle[3] - rectangle[1]),2))
    const leftTop = [centerx - squareLenght / 2, centery - squareLenght / 2]
    console.log(centerx, centery, squareLenght, leftTop)
    tf.util.assert(leftTop[0]<0 || leftTop[1]<0 || leftTop[1] + squareLenght > images.shape[0] || leftTop[0] + squareLenght > images.shape[1], 'Face is too close to the edge of the picture')
    // let img = fs.readFileSync('./public/images/wai.png')
    // let images = tf.node.decodeImage(img)
    let sub = []
    for(let j=0; j<images.shape[1]; j++){
        sub.push([225])  
    }
    let aImageArr = []
    for(let j=0; j<images.shape[0]; j++){
        aImageArr.push([...sub])  
    }
    aImageArr = tf.tensor(aImageArr,[images.shape[0], images.shape[1], 1],"int32")
    images = tf.concat([images, aImageArr], 2)

    // images = images.slice([rectangles[0][1],rectangles[0][0]], [rectangles[0][3] - rectangles[0][1], rectangles[0][2] - rectangles[0][0]])
    // console.log(images)
    // let squareLenght = Math.max((rectangles[0][2] - rectangles[0][0]), (rectangles[0][3] - rectangles[0][1]))
    // images = tf.image.resizeNearestNeighbor(images, [squareLenght, squareLenght])
    // console.log(images)

    //000275.jpg眼镜

    images = images.slice([leftTop[1], leftTop[0]], [squareLenght, squareLenght])
    images = images.as4D(1, images.shape[0], images.shape[1], 4)
    // images.print()
    //默认将鼻子当做预设人脸的中心点
    let eyeFactorx = 0.5 //预设的眼睛位置，眼睛到鼻子与眼睛到人脸矩形边界距离的x轴比值,可自定义。
    let eyeFactory = 0.5 //预设的眼睛位置，眼睛到鼻子与眼睛到人脸矩形边界距离的y轴比值,可自定义。
    let mouFactorx = 0.5 //预设的嘴巴位置，嘴巴端点到鼻子与嘴巴端点到人脸矩形边界距离的x轴比值,可自定义。
    let mouFactory = 0.5 //预设的嘴巴位置，嘴巴端点到鼻子与嘴巴端点到人脸矩形边界距离的y轴比值,可自定义。
    let offsetEyeX = (centerx - rectangle[0]) * eyeFactorx
    let offsetEyeY = (centery - rectangle[1]) * eyeFactory
    let offsetMouX = (centerx - rectangle[0]) * mouFactorx
    let offsetMouY = (centery - rectangle[1]) * mouFactory;
    [angle, scale, distance] = getAffineFactor(
        [[rectangle[5], rectangle[6]],[rectangle[7], rectangle[8]],[rectangle[9], rectangle[10]],[rectangle[11], rectangle[12]],[rectangle[13], rectangle[14]]],
        [[centerx-offsetEyeX, centery-offsetEyeY],[centerx+offsetEyeX, centery-offsetEyeY],[centerx, centery],[centerx-offsetMouX, centery+offsetMouY],[centerx+offsetMouX, centery+offsetMouY]])
    console.log("angle: ", angle, "scale: ", scale, "distance: ", distance)
    const basicAugmentation = ia.sequential([
        ia.affine({translatePercent: [0,0],rotate: -angle, scale:1}),
        ia.blur(1)
    ]);
    affineImages = await basicAugmentation.read({images:images})
    affineImages = affineImages.images
    console.log(affineImages)
    console.log(rectangle[1]-leftTop[1], rectangle[0]-leftTop[0], rectangle[3] - rectangle[1], rectangle[2] - rectangle[0])
    affineImages = affineImages.slice([0, rectangle[1]-leftTop[1], rectangle[0]-leftTop[0]],[1, rectangle[3] - rectangle[1], rectangle[2] - rectangle[0]])
    await tensorToFile('./public/images/zheng1.png', affineImages)
    let [t1, t2] = tf.split(affineImages, [3,1], 3)
    t1 = t1.reshape([t1.shape[1], t1.shape[2], 3])
    return t1
}
// affineImage()
exports.affineImage = affineImage
