const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const detectFace = require("./detectFace")
const facenet = require("./facenet")
const config = require("../configParameter/config.json")

function _load_appa(appa_dir, type){
    let dataPath = null
    let csvPath = null
    if(type=="train"){
        dataPath = "E:/tensorflow/age-gender-estimation-master/age_estimation/appa-real-release/train"
        csvPath = appa_dir + "/gt_avg_train.csv"
    }else if(type=="valid"){
        dataPath = "E:/tensorflow/age-gender-estimation-master/age_estimation/appa-real-release/valid"
        csvPath = appa_dir + "/gt_avg_valid.csv"
    }
    let data = fs.readFileSync(csvPath)
    data = ConvertToTable(data, dataPath)
    return data
}

function ConvertToTable(data, dataPath) {
    data = data.toString();
    let table = new Array();
    let rows = new Array();
    // console.log(data)
    rows = data.split("\n");
    for (let i = 1; i < rows.length; i++) {
        let r = rows[i].split(",")
        table.push([dataPath+"/"+r[0], Math.floor(r[2])]);
    }
    return table
}

function makeIterator() {
    let index = 0;
    let datas =  _load_appa("./public/appacsv", "train")
    //console.log(datas)  
    //datas = datas.slice(0, datas.length*config.trainParam.useTraDataScale)
    const iterator = {
        next: () => {
            let result;
            let i = index * config.trainParam.batch_size
            let length = Math.min(config.trainParam.batch_size, (datas.length - i))
            console.log("train ", index, " batch size: ", length)
            let batch_inputs = []
            let y = []
            for(let j=0; j<length; j++){
                [image_path, age] = datas[i+j]
                //console.log(age)
                let bias = Math.floor(Math.random()*2+0.5)
                //console.log(bias)
                age += bias
                age = age<0 ? 0 : age 
                age = age>100 ? 100 : age
                //console.log(age)
                let one_hot = new Array(101).fill(0)
                one_hot[age] = 1
                //console.log(one_hot)
                y.push(one_hot)
                let image = fs.readFileSync(image_path)
                let imgTensor = tf.node.decodeImage(image)
                let threshold = [0.6,0.6,0.7]
                console.log("loading", image_path)
                let rectangles = detectFace.detectFace(imgTensor, threshold, mtcnnModel[0], mtcnnModel[1], mtcnnModel[2])
                if(rectangles.length>1){
                    let maxArea = 0
                    let idx = 0
                    for(let k=0; k<rectangles.length; k++){
                        if((rectangles[k][3]-rectangles[k][1])*(rectangles[k][2]-rectangles[k][0])>maxArea){
                            maxArea = (rectangles[k][3]-rectangles[k][1])*(rectangles[k][2]-rectangles[k][0])
                            idx = k
                        }
                    }
                    rectangles = [rectangles[idx]]
                }
                //console.log(rectangles)
                imgTensor = imgTensor.arraySync()
                let imgTensorFace = []
                imgTensor.map((val,index)=>{
                    if(index>rectangles[0][1]&&index<=rectangles[0][3]){
                        imgTensorFace.push(val.slice(rectangles[0][0],rectangles[0][2]))
                    }
                })
                imgTensorFace = tf.image.resizeBilinear(tf.tensor(imgTensorFace,[rectangles[0][3]-rectangles[0][1],rectangles[0][2]-rectangles[0][0],3]),[299,299])
                imgTensorFace = facenet.prewhiten(imgTensorFace)
                imgTensorFace = imgTensorFace.reshape([1,299,299,3])
                //console.log(imgTensorFace)
                batch_inputs.push(imgTensorFace)
            }
            // console.log(batch_inputs.length)
            // console.log(y)
            if (datas.length - i < config.trainParam.batch_size) {
                result = {value: {xs: tf.concat(batch_inputs, 0), ys: tf.tensor(y)}, done: true};
                return result;
            }
            index++;
            return {value: {xs: tf.concat(batch_inputs, 0), ys: tf.tensor(y)}, done: false};
        }
    } 
    return iterator;
}

function makeIterator1() {
    let index = 0;
    let datas =  _load_appa("./public/appacsv", "valid")
    //console.log(datas)  
    //datas = datas.slice(0, datas.length*config.trainParam.useTraDataScale)
    const iterator = {
        next: () => {
            let result;
            let i = index * config.trainParam.batch_size
            let length = Math.min(config.trainParam.batch_size, (datas.length - i))
            console.log("valid ", index, " batch size: ", length)
            let batch_inputs = []
            let y = []
            for(let j=0; j<length; j++){
                [image_path, age] = datas[i+j]
                //console.log(age)
                let one_hot = new Array(101).fill(0)
                one_hot[age] = 1
                //console.log(one_hot)
                y.push(one_hot)
                let image = fs.readFileSync(image_path)
                let imgTensor = tf.node.decodeImage(image)
                let threshold = [0.6,0.6,0.7]
                console.log("loading", image_path)
                let rectangles = detectFace.detectFace(imgTensor, threshold, mtcnnModel[0], mtcnnModel[1], mtcnnModel[2])
                if(rectangles.length>1){
                    let maxArea = 0
                    let idx = 0
                    for(let k=0; k<rectangles.length; k++){
                        if((rectangles[k][3]-rectangles[k][1])*(rectangles[k][2]-rectangles[k][0])>maxArea){
                            maxArea = (rectangles[k][3]-rectangles[k][1])*(rectangles[k][2]-rectangles[k][0])
                            idx = k
                        }
                    }
                    rectangles = [rectangles[idx]]
                }
                //console.log(rectangles)
                imgTensor = imgTensor.arraySync()
                let imgTensorFace = []
                imgTensor.map((val,index)=>{
                    if(index>rectangles[0][1]&&index<=rectangles[0][3]){
                        imgTensorFace.push(val.slice(rectangles[0][0],rectangles[0][2]))
                    }
                })
                imgTensorFace = tf.image.resizeBilinear(tf.tensor(imgTensorFace,[rectangles[0][3]-rectangles[0][1],rectangles[0][2]-rectangles[0][0],3]),[299,299])
                imgTensorFace = facenet.prewhiten(imgTensorFace)
                imgTensorFace = imgTensorFace.reshape([1,299,299,3])
                //console.log(imgTensorFace)
                batch_inputs.push(imgTensorFace)
            }
            // console.log(batch_inputs.length)
            //console.log(y)
            if (datas.length - i < config.trainParam.batch_size) {
                result = {value: {xs: tf.concat(batch_inputs, 0), ys: tf.tensor(y)}, done: true};
                return result;
            }
            index++;
            return {value: {xs: tf.concat(batch_inputs, 0), ys: tf.tensor(y)}, done: false};
        }
    } 
    return iterator;
}


async function GetData(usage){
    mtcnnModel = await detectFace.loadModel(config.modelPath.pModelPath, config.modelPath.rModelPath, config.modelPath.oModelPath)
    if(usage=='train'){
        const ds = tf.data.generator(makeIterator);
        return ds
        // let ds = await makeIterator()
        // ds = ds.next()
        // console.log("ha",ds.value)
        // ds.value.ys.print()
    }else{
        const ds = tf.data.generator(makeIterator1);
        return ds
        // let ds = await makeIterator1()
        // ds = ds.next()
        // console.log("ha1",ds.value)
        // ds.value.ys.print()
    }
}

// GetData("vaild")
exports.GetData = GetData