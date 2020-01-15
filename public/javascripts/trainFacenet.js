
const tf = require("@tensorflow/tfjs-node");
const facenet = require("./facenet")
const generator = require("./dataGenerator")
const tfjs_core_1 = require("@tensorflow/tfjs-core");
const tfc = require("@tensorflow/tfjs-core")
const config = require("../configParameter/config.json")
// const modelPath = "E:/tensorflow/facenet-master/models/trainedFacenet/model.json"
// const alpha = 0.2 //triplet loss参数
// const num_train_samples = 196998 //celeA数据集总数202599 - 未识别图片数5,600 = 196999 再减去最后一张
// const num_lfw_valid_samples = 2185 //测试集数据总量
// const useValDataScale = 0.1 //使用验证数据集的比例
// const useTraDataScale = 0.01 //使用训练数据集的比例
// const batch_size = 32 //批次大小
// const epochs = 5 //总迭代次数

//如果在模型中定义tf的计算会导致内存溢出所以定义函数来防止这种情况
function LambdaFunction(input, scale){
    return tf.add(tf.mul(input[1],tf.scalar(scale)),input[0])
}

async function loadOriginalFacenetModel(modelPath){
    class Lambda017 extends tf.layers.Layer {
        constructor() {
            super({});
        }
        // In this case, the output is a scalar.
        computeOutputShape(input) { return input[0]; }
        // call() is where we do the computation.
        call(input,kwargs) {
            return LambdaFunction(input, 0.17)
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
        computeOutputShape(input) { return input[0]; }
        // call() is where we do the computation.
        call(input,kwargs) {
            return LambdaFunction(input, 0.1)
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
        computeOutputShape(input) { return input[0]; }    
        // call() is where we do the computation.
        call(input,kwargs) {
            return LambdaFunction(input, 0.2)
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
        computeOutputShape(input) { return input[0]; }    
        // call() is where we do the computation.
        call(input,kwargs) {
            return LambdaFunction(input, 1)
        }       
        // Every layer needs a unique name.
        getClassName() { return 'Lambda'; }
    }
    Lambda1.className = 'Lambda1'
    
    class LambdaL2Normalize extends tf.layers.Layer {        
        constructor() {
            super({});
        }     
        // In this case, the output is a scalar.
        computeOutputShape(input) { return input; }    
        // call() is where we do the computation.
        call(input,kwargs) {
            return facenet.l2_normalize(input[0])
        }       
        // Every layer needs a unique name.
        getClassName() { return 'LambdaL2Normalize'; }
    }
    LambdaL2Normalize.className = 'LambdaL2Normalize'

    //注册自定义层
    tf.serialization.registerClass(Lambda017);
    tf.serialization.registerClass(Lambda01);
    tf.serialization.registerClass(Lambda02);
    tf.serialization.registerClass(Lambda1);
    tf.serialization.registerClass(LambdaL2Normalize);

    //加载模型
    const model = await tf.loadLayersModel('file://'+modelPath);
    console.log(model.summary())
    // console.log(tf.zeros([1,160,160,3]).dtype)
    // let result = model.predict([tf.zeros([1,160,160,3], "float32"), tf.zeros([1,160,160,3], "float32"), tf.zeros([1,160,160,3], "float32")])
    // console.log(result.print())
    return model
}

//定义三元组损失
function triplet_loss(yTrue, yPred){
    return tfjs_core_1.tidy(function(){
        let [a_pred, p_pred, n_pred] = tf.split(yPred,3,1)
        let positive_distance = tfc.square(tfc.norm(tfc.sub(a_pred, p_pred), 'euclidean', -1))
        let negative_distance = tfc.square(tfc.norm(tfc.sub(a_pred, n_pred), 'euclidean', -1))
        let loss = tfc.mean(tfc.maximum(0.0, tfc.add(tfc.sub(positive_distance, negative_distance), config.trainParam.alpha)))
        console.log("hi triplet", loss.print())
        return loss
    })
}

async function train(){
    //加载模型
    let originalFacenetModel = await loadOriginalFacenetModel(config.modelPath.pretrainedFacenetModel)
    //添加优化器和损失函数
    originalFacenetModel.compile({optimizer:tf.train.sgd(1e-5), loss:triplet_loss, metrics: ['acc']})

    //开始微调模型
    originalFacenetModel.fitDataset(await generator.GetData("train"), {
        batchesPerEpoch: Math.ceil(config.trainParam.num_train_samples * config.trainParam.useTraDataScale / config.trainParam.batch_size),
        epochs: config.trainParam.epochs,
        verbose: 1,
        callbacks: {
            onTrainBegin: async () => {
                console.log("onTrainBegin")
              },
              onTrainEnd: async (epoch, logs) => {
                console.log("onTrainEnd" + epoch + JSON.stringify(logs))
              },
              onEpochBegin: async (epoch, logs) => {
                console.log("onEpochBegin" + epoch + JSON.stringify(logs))
              },
              onEpochEnd: async (epoch, logs) => {
                console.log("onEpochEnd" + epoch + JSON.stringify(logs))
              },
              onBatchBegin: async (epoch, logs) => {
                console.log("onBatchBegin" + epoch + JSON.stringify(logs))
              },
              onBatchEnd: async (epoch, logs) => {
                console.log("onBatchEnd" + epoch + JSON.stringify(logs))
              }
        },
        validationData: await generator.GetData("validation"),
        validationBatches: Math.ceil(config.trainParam.num_lfw_valid_samples * config.trainParam.useValDataScale / config.trainParam.batch_size)
    })
}

train()
// triplet_loss(null, tf.zeros([384]))
// let result = originalFacenetModel.predict([tf.zeros([1,160,160,3]), tf.zeros([1,160,160,3]), tf.zeros([1,160,160,3])])
// console.log(result)