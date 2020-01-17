const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const detectFace = require("./detectFace")
const facenet = require("./facenet")
const readLine = require("readline");
const config = require("../configParameter/config.json") //从配置文件导入训练相关参数
// const lfw_folder = 'E:/tensorflow/facenet-master/data/lfw' //验证lfw数据集地址
// const CelebAFolder = 'E:/tensorflow/facenet-master/data/img_align_celeba'
// const identity_annot_filename = 'E:/tensorflow/facenet-master/data/identity_CelebA.txt'
// const excludeFileName = 'E:/tensorflow/facenet-master/data/exclude.txt'
// const num_train_samples = 196998 //celeA数据集总数202599 - 未识别图片数5,600 = 196999 再减去最后一张
// const embedding_size = 128  //生成的人脸向量大小
// const useValDataScale = 0.1 //使用验证数据集的比例
// const useTraDataScale = 0.01 //使用训练数据集的比例
// const batch_size = config.trainParam.batch_size //批次大小
// const pModelPath = './public/model/Pnet/model.json'
// const rModelPath = './public/model/Rnet/model.json'
// const oModelPath = './public/model/Onet/model.json'
let mtcnnModel = []

//文件逐行读取函数
//input parameters: 
//  String modelPath 网络拓扑结构json文件地址
//  Function cb 回调函数
function readFileToArr(fReadName, cb) {

    var arr = [];
    var readObj = readLine.createInterface({
        input: fs.createReadStream(fReadName)
    });

    readObj.on('line', function (line) {
        arr.push(line);
    });
    readObj.on('close', function () {
        cb(arr);
    });
}

//从指定文件中读取celebA的图片名并标签对应关系
//input parameters: 
//  String celebAFileName celebA标记文件路径
//output:
//  Array[] Array[4] [ids, images.sort(), image2id, id2images] ids代表每张图片对应的id、images代表每张图片的文件名、image2id代表了文件名到id的映射、id2images代表了id到文件名的映射
async function get_data_stats(celebAFileName){
    let lines = []
    let excludes = []
    await new Promise(resolve => {
        readFileToArr(celebAFileName, function (arr) {
            resolve('resolved');
            lines = arr;
        })
    })
    await new Promise(resolve => {
        readFileToArr(config.trainData.excludeFileName, function (arr) {
            resolve('resolved');
            excludes = arr;
        })
    })
    ids = []
    images = []
    image2id = {}
    id2images = {}
    
    for (line of lines){
        if(line.length > 0){
            let tokens = line.split(' ')
            let image_name = tokens[0]
            if (image_name != '202599.jpg' && !excludes.includes(image_name)){
                let id = tokens[1]
                ids.push(id)
                images.push(image_name)
                image2id[image_name] = id
                if(id2images.hasOwnProperty(id)){
                    id2images[id].push(image_name)
                }else{
                    id2images[id] = [image_name]
                }
            }
                
        }
    }

    return [ids, images.sort(), image2id, id2images]

}

//从数据集中随机的生成满足需要的triplet-loss数据元组
//input parameters: 
//  null
//output:
//  生成形如{'a': a_image, 'p': p_image, 'n': n_image}的数据集对象数组
async function get_random_triplets(){
    [ids, images, image2id, id2images] = await get_data_stats(config.trainData.identity_annot_filename)
    images = images.slice(0, config.trainParam.num_train_samples)
    let data_set = []


    for(let i=0; i<config.trainParam.num_train_samples; i++){
        let a_image = null, p_image = null, n_image = null, a_id = null

        while(true){
            a_image = images[Math.floor((Math.random()*images.length))]
            a_id = image2id[a_image]
            if(id2images[a_id].length >= 2){
                break
            }
        }

        while(true){
            // console.log(id2images[a_id])
            p_image = id2images[a_id][Math.floor((Math.random()*id2images[a_id].length))]
            if(p_image != a_image){
                break
            }
        }

        while(true){
            n_image = images[Math.floor((Math.random()*images.length))]
            let n_id = image2id[n_image]
            if(n_id != a_id){
                break
            }
        }

        data_set.push({'a': a_image, 'p': p_image, 'n': n_image})
    }
    // console.log(data_set)
    return data_set
}

//构建训练集数据生成迭代器
//input parameters: 
//  null
//output:
//  Object {value:{xs:一个批次的训练数据,ys:对应的标签}, done: true/flase 代表了迭代器是否结束}
async function makeIterator() {
    let index = 0;
    let datas =  await get_random_triplets() 
    datas = datas.slice(0, datas.length*config.trainParam.useTraDataScale)
    const iterator = {
        next: () => {
            let result;
            let i = index * config.trainParam.batch_size
            let length = Math.min(config.trainParam.batch_size, (datas.length - i))
            console.log("train ", index, " batch size: ", length)
            let batch_dummy_target = tf.zeros([length, config.trainParam.embedding_size * 3], "float32")//由于triplet-loss训练不需要标签，所以batch_dummy_target在这里可以任意初始化
            let batch_inputs = [[],[],[]]
            for(let j=0; j<length; j++){
                console.log(j)
                sample = datas[i + j];
                ['a', 'p', 'n'].map((val,index)=>{
                    let image_name = sample[val]
                    let filename = config.trainData.CelebAFolder + '/' + image_name
                    let img = fs.readFileSync(filename)
                    let imgTensor = tf.node.decodeImage(img)
                    let threshold = [0.6,0.6,0.7]
                    console.log("loading", val, filename)
                    let rectangles = detectFace.detectFace(imgTensor, threshold, mtcnnModel[0], mtcnnModel[1], mtcnnModel[2])
                    imgTensor = imgTensor.arraySync()
                    let imgTensorFace = []
                    imgTensor.map((val,index)=>{
                        if(index>rectangles[0][1]&&index<=rectangles[0][3]){
                            imgTensorFace.push(val.slice(rectangles[0][0],rectangles[0][2]))
                        }
                    })
                    imgTensorFace = tf.image.resizeBilinear(tf.tensor(imgTensorFace,[rectangles[0][3]-rectangles[0][1],rectangles[0][2]-rectangles[0][0],3]),[160,160])
                    imgTensorFace = facenet.prewhiten(imgTensorFace)
                    console.log(imgTensorFace)
                    imgTensorFace.print()
                    batch_inputs[index].push(imgTensorFace.arraySync())
                })
            }
            if (datas.length - i < config.trainParam.batch_size) {
                result = {value: {xs: [tf.tensor(batch_inputs[0], [batch_inputs[0].length, 160, 160, 3]), tf.tensor(batch_inputs[1], [batch_inputs[1].length, 160, 160, 3]), tf.tensor(batch_inputs[2], [batch_inputs[2].length, 160, 160, 3])], ys: batch_dummy_target}, done: true};
                index++;
                return result;
            }
            return {value: {xs: [tf.tensor(batch_inputs[0], [batch_inputs[0].length, 160, 160, 3]), tf.tensor(batch_inputs[1], [batch_inputs[1].length, 160, 160, 3]), tf.tensor(batch_inputs[2], [batch_inputs[2].length, 160, 160, 3])], ys: batch_dummy_target}, done: false};
        }
    } 
    return iterator;
}

//构建验证集数据生成迭代器
//input parameters: 
//  null
//output:
//  Object {value:{xs:一个批次的验证数据,ys:对应的标签}, done: true/flase 代表了迭代器是否结束}
async function makeIterator1() {
    let index = 0;
    const lfw_val_triplets = require(config.trainData.lfw_val_triplets)
    let datas =  lfw_val_triplets.slice(0, lfw_val_triplets.length*config.trainParam.useValDataScale) 
    const iterator = {
        next: () => {
            let result;
            let i = index * config.trainParam.batch_size
            let length = Math.min(config.trainParam.batch_size, datas.length - i)
            console.log("validation ", index, " batch size: ", length)
            let batch_dummy_target = tf.zeros([length, config.trainParam.embedding_size * 3], "float32")
            let batch_inputs = [[],[],[]]
            for(let j=0; j<length; j++){
                console.log(j)
                sample = datas[i + j];
                ['a', 'p', 'n'].map((val,index)=>{
                    let image_name = sample[val]
                    let filename = config.trainData.lfw_folder + '/' + image_name
                    let img = fs.readFileSync(filename)
                    let imgTensor = tf.node.decodeImage(img)
                    let threshold = [0.6,0.6,0.7]
                    console.log("loading", val, filename)
                    let rectangles = detectFace.detectFace(imgTensor, threshold, mtcnnModel[0], mtcnnModel[1], mtcnnModel[2])
                    imgTensor = imgTensor.arraySync()
                    let imgTensorFace = []
                    imgTensor.map((val,index)=>{
                        if(index>rectangles[0][1]&&index<=rectangles[0][3]){
                            imgTensorFace.push(val.slice(rectangles[0][0],rectangles[0][2]))
                        }
                    })
                    imgTensorFace = tf.image.resizeBilinear(tf.tensor(imgTensorFace,[rectangles[0][3]-rectangles[0][1],rectangles[0][2]-rectangles[0][0],3]),[160,160])
                    imgTensorFace = facenet.prewhiten(imgTensorFace)
                    batch_inputs[index].push(imgTensorFace.arraySync())
                })
            }
            if (datas.length - i < config.trainParam.batch_size) {
                result = {value: {xs: [tf.tensor(batch_inputs[0], [batch_inputs[0].length, 160, 160, 3]), tf.tensor(batch_inputs[1], [batch_inputs[1].length, 160, 160, 3]), tf.tensor(batch_inputs[2], [batch_inputs[2].length, 160, 160, 3])], ys: batch_dummy_target}, done: true};
                index++;
                return result;
            }
            return {value: {xs: [tf.tensor(batch_inputs[0], [batch_inputs[0].length, 160, 160, 3]), tf.tensor(batch_inputs[1], [batch_inputs[1].length, 160, 160, 3]), tf.tensor(batch_inputs[2], [batch_inputs[2].length, 160, 160, 3])], ys: batch_dummy_target}, done: false};
        }
    } 
    return iterator;
}

//通过传入的usage来决定是返回训练集生成器还是验证集生成器
async function GetData(usage){
    mtcnnModel = await detectFace.loadModel(config.modelPath.pModelPath, config.modelPath.rModelPath, config.modelPath.oModelPath)
    if(usage=='train'){
        const ds = tf.data.generator(makeIterator);
        return ds
    }else{
        const ds = tf.data.generator(makeIterator1);
        return ds
    }
}

//GetData('train')
exports.GetData = GetData