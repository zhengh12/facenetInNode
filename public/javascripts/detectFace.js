const tf = require("@tensorflow/tfjs-node");
const toolMatrix = require("./toolMatrix")

//根据模型路径加载mtcnn的三个网络模型
//input parameters: 
//  String pModelPath mtcnnPnet的json文件路径
//  String rModelPath mtcnnRnet的json文件路径
//  String oModelPath mtcnnOnet的json文件路径
//output:
//  tf.LayersModel Array[3] [Pnet, Rnet, Onet] mtcnn的Pnet，Rnet，Onet网络模型
async function loadModel(pModelPath, rModelPath, oModelPath){
    const Pnet = await tf.loadLayersModel('file://'+pModelPath);
    const Rnet = await tf.loadLayersModel('file://'+rModelPath);
    const Onet = await tf.loadLayersModel('file://'+oModelPath);
    return [Pnet, Rnet, Onet]
}

//利用mtcnn进行人脸检测
//该函数及其工具toolMatrix仅利用少数tfjs的api编写,所以速度还可以改善
//input parameters: 
//  tf.Tensor imgarrs 原图像的数据以张量形式表示
//  Array[3] threshold 人脸检测过程中用到的参数
//  tf.LayersModel Pnet mtcnn快速扫描人脸区域的Pnet
//  tf.LayersModel Rnet mtcnn精确判断人脸区域的Rnet
//  tf.LayersModel Pnet mtcnn定位人脸地标位置的Onet
//output:
//  Number array[n][15] rectangles 得到零到多个包含矩形框左上右下两点x1, y1, x2, y2并人脸置信得分和人脸五个地标点位置的人脸区域数据数组
function detectFace(imgarrs, threshold, Pnet, Rnet, Onet){
    // console.time('run 12net1 time')
    let caffe_img = tf.div(imgarrs.sub(tf.scalar(127.5)), tf.scalar(127.5))
    const origin_h = caffe_img.shape[0]
    const origin_w = caffe_img.shape[1]
    let scales = toolMatrix.calculateScales(origin_h, origin_w)//获得
    caffe_img = caffe_img.arraySync()
    let out = []

    // 将图片转化为图形金字塔输入PNet中
    scales.map((scale,index)=>{
        // console.time('run 12net1 '+index+' time')
        let hs = Math.floor(origin_h * scale)
        let ws = Math.floor(origin_w * scale)
        let imgTensors = tf.image.resizeBilinear(imgarrs,[hs,ws])
        let scale_imgs = tf.div(imgTensors.sub(tf.scalar(127.5)), tf.scalar(127.5))
        //console.log(scale_imgs.length,scale_imgs[0].length)
        // console.log(scale.toString()+":",scale_imgs[10][10])
        let input = scale_imgs.reshape([1,hs,ws,3])
        let ouput = Pnet.predict(input)

        let output = ouput.map( val=>{
            return val.arraySync()
        })
        //console.log(scale.toString()+":",output[0][0][0][0])
        out.push(output)
        // console.timeEnd('run 12net1 '+index+' time')
        // console.log(out)
    })

    // let arrs = [[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]],[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]]]
    // arrs = arrs.map(val=>{
    //     return val.map(val=>{
    //         return val[1]
    //     })
    // })
    // console.log(arrs)
    // console.timeEnd('run 12net1 time')
    // console.time('run 12net2 time')

    let rectangles = []
    out.map((val,index)=>{
        // console.time('run 12net2 '+index+' time')
        let cls_prob = val[0][0].map(val=>{
            return val.map(val=>{
                return val[1]
            })
        })
        let roi = val[1][0]
        let out_h = cls_prob.length
        let out_w = cls_prob[0].length
        let out_side = Math.max(out_h, out_w)
        //cls_prob = np.swapaxes(cls_prob, 0, 1)
        //roi = np.swapaxes(roi, 0, 2)
        cls_prob = Array(cls_prob[0].length).fill(null).map((val,index1) => {
            return Array(cls_prob.length).fill(null).map((val,index2)=>{
                return cls_prob[index2][index1]
            })
        })//数组按0,1轴转置
        roi = Array(roi[0][0].length).fill(null).map((val,index0) => {
            return Array(roi[0].length).fill(null).map((val,index1)=>{
                return Array(roi.length).fill(null).map((val,index2)=>{
                    return roi[index2][index1][index0]
                })
            })
        })//数组按0,2轴转置
        //console.log(cls_prob.length,cls_prob[0].length)
        //console.log(roi.length,roi[0].length,roi[0][0].length)
        let rectangle = toolMatrix.detect_face_12net(cls_prob, roi, out_side, 1 / scales[index], origin_w, origin_h, threshold[0])
        rectangle.map(val=>{
            rectangles.push(val)
        }) 
        // console.timeEnd('run 12net2 '+index+' time')   
    })

    //tf.image.nonMaxSuppression这个方法也可以
    
    rectangles = toolMatrix.NMS(rectangles, 0.7, 'iou')
    // console.timeEnd('run 12net2 time')
    // console.log('rectangles12',rectangles,rectangles.length)
    // return rectangles
    // console.time('run 24net time')
    if (rectangles.length === 0){
        return rectangles
    }

    let crop_number = 0
    out = []
    let predict_24_batch = []
    //rectangles=[[153.0, 445.0, 170.0, 462.0, 0.6271253228187561],[189.0, 513.0, 206.0, 529.0, 0.6269974112510681]]
    rectangles.map(val=>{
        //console.log(caffe_img)
        let crop_img = caffe_img.map(vals=>{
            // console.log("val:",vals)
            // console.log("vals:",vals.slice(val[0],val[2]+1))
            return vals.slice(val[0],val[2])
        }).slice(val[1],val[3])
        // console.log(crop_img)
        // console.log(crop_img.length,crop_img[0].length)
        let input = tf.tensor(crop_img,[crop_img.length,crop_img[0].length,3])
        // console.log(input)
        let scale_img = tf.image.resizeBilinear(input,[24,24])
        //console.log("scale_img",scale_img)
        scale_img = scale_img.arraySync()
        //console.log("scale_imgs",scale_img[0][0])
        predict_24_batch.push(scale_img)
    })
    predict_24_batch = tf.tensor(predict_24_batch)
    out = Rnet.predict(predict_24_batch)
    let cls_prob = out[0].arraySync()  
    let roi_prob = out[1].arraySync() 
    rectangles = toolMatrix.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])

    // console.log('rectangles24',rectangles,rectangles.length)
    // console.timeEnd('run 24net time')
    // console.time('run 48net time')
    if (rectangles.length === 0){
        return rectangles
    }

    let predict_batch = []
    rectangles.map(val=>{
        //console.log(caffe_img)
        let crop_img = caffe_img.map(vals=>{
            // console.log("val:",vals)
            // console.log("vals:",vals.slice(val[0],val[2]+1))
            return vals.slice(val[0],val[2])
        }).slice(val[1],val[3])
        // console.log(crop_img)
        // console.log(crop_img.length,crop_img[0].length)
        let input = tf.tensor(crop_img,[crop_img.length,crop_img[0].length,3])
        // console.log(input)
        let scale_img = tf.image.resizeBilinear(input,[48,48])
        //console.log("scale_img",scale_img)
        scale_img = scale_img.arraySync()
        //console.log("scale_imgs",scale_img[0][0])
        predict_batch.push(scale_img)
    })
    predict_batch = tf.tensor(predict_batch)
    let output = Onet.predict(predict_batch)
    // console.log(output,output.length)
    cls_prob = output[0].arraySync()
    // console.log(cls_prob)
    roi_prob = output[1].arraySync()
    let pts_prob = output[2].arraySync()
    rectangles = toolMatrix.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
    // console.timeEnd('run 48net time')
    return rectangles
}

exports.detectFace = detectFace
exports.loadModel = loadModel