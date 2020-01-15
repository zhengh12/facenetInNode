const kmeans = require('ml-kmeans');
const facenet = require("./facenet")
const fs = require("fs");
const tf = require("@tensorflow/tfjs-node");
const RandomForest = require('ml-random-forest') 
const detectFace = require("./detectFace")
const { agnes } = require('ml-hclust');
const clustering = require('density-clustering');
const snapList = require('snap-to-grid-clustering').snapList
const {Matrix, EigenvalueDecomposition, covariance} = require('ml-matrix');

//读到文件夹下所有文件的路径
function findFile(dataPath){
    let dirArr=[]
    let dir = fs.readdirSync(dataPath)
    dir.map(item=>{
        let stat = fs.lstatSync(dataPath + item)
        if (stat.isDirectory() === true) { 
            let subDirArr = []
            let subDir = fs.readdirSync(dataPath + item + '/')
            subDir.map(val=>{
                let stat = fs.lstatSync(dataPath + item + '/' + val)
                if(stat.isDirectory() === false){
                    subDirArr.push(dataPath + item + '/' + val)
                }
            })
            dirArr.push(subDirArr)
        }
    })
    return dirArr
}

//主文件夹下包含多个子文件夹，每个子文件夹作为一个人脸分类，一个子文件夹包含至少一张人脸图片
async function loadFiles(dataPath, FacenetModel, Pnet, Rnet, Onet){
    let dirArr = findFile(dataPath)
    let vectors = []
    for(let i=0; i<dirArr.length; i++){
        let subvector = []
        for(val of dirArr[i]){
            console.log("loading file: "+val)
            let vector = await facenet.faceVector(FacenetModel, val, Pnet, Rnet, Onet)
            vector = vector.arraySync()[0]
            // vector.push(i)
            // vector = vector.map((val,index)=>{
            //     return vector.slice(0,index).concat(vector.slice(index+1,vector.length)).concat([index.toString()])
            // })
            subvector.push(vector)
        }
        vectors.push(subvector)
    }
    console.log("loading file over")
    return vectors
}

//特征值加权降维
//128维向量特征加权卷积后的维度为 128-convolutionSize+1
//convectorsAvg: 每一类的类间平均值
function EigenvalueCon(vectors, convolutionSize){
    let real = findEigenvalue(vectors)
    let convectorsAvg = []
    let convectorsAll = []
    for(subVectors of vectors){
        let subConvectorsAll = []
        let avgVector = tf.scalar(0)
        for(val of subVectors){
            let convector = []
            for(let i=0; i<val.length; i++){
                let EigenvalueSum = real.slice(i, i+convolutionSize).reduce(function (a, b) { return a + b;})
                let res = i+convolutionSize<=val.length ? val.slice(i, i+convolutionSize).reduce(function (a, b, index) { return a + b * real[i+index] }, 0)/EigenvalueSum : null
                if(res !== null){
                    convector.push(res)
                }    
            }
            
            // let out = tf.tensor(val)
            // let out1 = tf.tensor([...convector,...val.slice(val.length-convolutionSize,val.length)])
            // let dist = tf.sqrt(tf.sum(tf.squaredDifference(out,out1)))
            // dist.print()
            avgVector = tf.add(avgVector, convector)
            subConvectorsAll.push(convector)
        }
        avgVector = tf.div(avgVector,tf.scalar(subVectors.length))
        convectorsAvg.push(avgVector.arraySync())
        convectorsAll.push(subConvectorsAll)
    }
    return[convectorsAll, convectorsAvg]
}

//平均值降维
//128维平均值卷积后的维度为 128-convolutionSize+1
//convectorsAvg: 每一类的类间平均值
function con(vectors, convolutionSize){
    let convectorsAvg = []
    let convectorsAll = []
    for(subVectors of vectors){
        let subConvectorsAll = []
        let avgVector = tf.scalar(0)
        for(val of subVectors){
            let convector = []
            for(let i=0; i<val.length; i++){
                let res = i+convolutionSize<=val.length ? val.slice(i, i+convolutionSize).reduce(function (a, b, index) { return a + b })/convolutionSize : null
                if(res !== null){
                    convector.push(res)
                }    
            }
            
            // let out = tf.tensor(val)
            // let out1 = tf.tensor([...convector,...val.slice(val.length-convolutionSize,val.length)])
            // let dist = tf.sqrt(tf.sum(tf.squaredDifference(out,out1)))
            // dist.print()
            avgVector = tf.add(avgVector, convector)
            subConvectorsAll.push(convector)
        }
        avgVector = tf.div(avgVector,tf.scalar(subVectors.length))
        convectorsAvg.push(avgVector.arraySync())
        convectorsAll.push(subConvectorsAll)
    }
    return[convectorsAll, convectorsAvg]
}

//从已经分好类的目录算出测试数据实际上应该被分在哪一类
function getBeforePredictResult(trainDatapath, predictDatapath, ans){
    let dir = findFile(trainDatapath)
    let realClass = []
    for(let i=0; i<dir.length; i++){
        let str = dir[i][0].split("/")
        realClass.push([str[str.length-2], ans[i]])
    }
    // console.log("realClass",realClass)
    let predictResult = []
    let predictDir = findFile(predictDatapath)
    for(let i=0; i<predictDir.length; i++){
        for(val of predictDir[i]){
            let str = val.split("/")
            for(val of realClass){
                if(str[str.length-2]==val[0]){
                    predictResult.push(val[1])
                }
            }
        }
    }
    return predictResult
}

function sortToIndex(originalArr){
    originalArr = originalArr.map(function(item,index){
        return {'key':index,'value':item}
    }) 
    let indexArr = originalArr.sort(function(a,b){
        return b.value - a.value
    }).map(function(item){
        return item.key
    })
    return indexArr
}

//PCA主成分分析
//k为主成分分析中需要保留的前k维特征值 k需要小于等于当前特征维数
function PCA(vectorsAll, k){
    let tensor = tf.tensor(vectorsAll)
    tensor = tf.sub(tensor, tf.mean(tensor,0)).arraySync()
    let vectorMatrix = new Matrix(tensor)
    let covMatrix = covariance(vectorMatrix) //协方差矩阵
    //console.log(covMatrix)
    let EigenvalueMatrix = new EigenvalueDecomposition(covMatrix); //特征类
    let real = EigenvalueMatrix.realEigenvalues //特征数组
    let EigenvectorMatrix = EigenvalueMatrix.eigenvectorMatrix
    let indexArr = sortToIndex(real)
    if(k>indexArr.length){
        console.log("K value is too large! Please select a smaller value")
        return null
    }
    EigenvectorMatrix = indexArr.slice(0, k).map(val=>{
        return EigenvectorMatrix.getRowVector(val).to1DArray()
    })
    EigenvectorMatrix = new Matrix(EigenvectorMatrix)
    let PCAVectors = vectorMatrix.mmul(EigenvectorMatrix.transpose()).to2DArray()
    return PCAVectors
}

async function createTree(){
    const trainDatapath = './public/images/RandomForestTrainData1/'
    const predictDatapath = './public/images/RandomForestPredictData/'
    const modelPath = "./public/model/Facenet1/model.json"
    const pModelPath = './public/model/Pnet/model.json'
    const rModelPath = './public/model/Rnet/model.json'
    const oModelPath = './public/model/Onet/model.json'
    const FacenetModel = await facenet.loadFacenetModel(modelPath)
    const mtcnnModel = await detectFace.loadModel(pModelPath, rModelPath, oModelPath)
    
    //获取训练数据和测试数据
    let trainVectors = await loadFiles(trainDatapath, FacenetModel, mtcnnModel[0], mtcnnModel[1], mtcnnModel[2])
    let predictVectors = await loadFiles(predictDatapath, FacenetModel, mtcnnModel[0], mtcnnModel[1], mtcnnModel[2])   
    
    let vectorsAll = []
    let vectorsAvg = []
    for(subVectors of trainVectors){
        let sum = tf.scalar(0)
        for(val of subVectors){
            vectorsAll.push(val)
            sum = sum.add(tf.tensor(val))
        }
        vectorsAvg.push(tf.div(sum,tf.scalar(subVectors.length)).arraySync())
    } 
    const convolutionSize = 4
    let lowVectorsAll = vectorsAll //采用全部向量
    let lowVectors = vectorsAvg //采用类内平均向量
    let Class = []
    let flag = 0
    for(let k=0; k<120; k++){
        lowVectors = PCA(lowVectors, 128-k*1)
        lowVectorsAll = PCA(lowVectorsAll, 128-k*1)
        console.log(lowVectors.length, lowVectors[0].length)
        let scaleAll = 0
        Class = []
        for(let i=0; i<lowVectors.length; i++){
            let scale = 0
            for(let j=0; j<lowVectors.length; j++){
                let dis = tf.sqrt(tf.sum(tf.squaredDifference(tf.tensor(lowVectors[i]),tf.tensor(lowVectors[j])))).arraySync()
                scale = dis<1 ? scale+1 : scale 
            }
            scale = scale/lowVectors.length
            // console.log(scale)
            if(scale < 0.5){
                Class.push(0)
            }else{
                Class.push(1)
            }
            scaleAll = scaleAll + scale
        }
        if(scaleAll/lowVectors.length > 0.5){
            flag = k
            break
        }
    }
    console.log("yeye", Class)

    let convectorsAvg = lowVectors
    let center = []
    let max = 0
    for(let m=0; m<convectorsAvg.length; m++){
        for(let n=0; n<convectorsAvg.length; n++){
            let dis = tf.sqrt(tf.sum(tf.squaredDifference(tf.tensor(convectorsAvg[m]),tf.tensor(convectorsAvg[n])))).arraySync()
            center = dis>max ? [convectorsAvg[m],convectorsAvg[n]] : center
            max = dis>max ? dis : max
        }
    }

    ans = kmeans(convectorsAvg, 2, { initialization: center, maxIterations:1000})
    console.log(ans)
    
    let predictions = []
    for(let i=0; i<trainVectors.length; i++){
        for(let j=0; j<trainVectors[i].length; j++){
            predictions.push(ans.clusters[i])
        }
    }

    const options = {
        seed: 3,
        maxFeatures: 0.8,
        replacement: true,
        nEstimators: 100,
        useSampleBagging: true
    };
    let classifer = new RandomForest.RandomForestClassifier(options);
    classifer.train(lowVectorsAll,predictions)

    let beforePredictResult = getBeforePredictResult(trainDatapath, predictDatapath, ans.clusters)
    console.log(beforePredictResult)
    let pVectors = []
    for(subVectors of predictVectors){
        for(val of subVectors){
            pVectors.push(val)
        }
    } 
    //利用分类器进行分类
    for(let k=0; k<flag+1; k++){
        pVectors = PCA(pVectors, 128-k*1) 
    }
    let predictConVectors = pVectors
    let result = classifer.predict(predictConVectors);
    let accuracy = result.reduce((sum, val, index) =>{ let b = val===beforePredictResult[index] ? 1 : 0; return sum + b}, 0) / result.length
    console.log("classifer accuracy: ", "Class1: ", ans.centroids[0].size, "Class2: ", ans.centroids[1].size, accuracy)

    vvvvv
    // let vectorsAll = []
    // for(subVectors of trainVectors){
    //     let sum = tf.scalar(0)
    //     for(val of subVectors){
    //         sum = tf.add(tf.tensor(val),sum)
    //     }
    //     vectorsAll.push(tf.div(sum,tf.scalar(subVectors.length)).arraySync())
    // } 
    // console.log(vectorsAll.length, "....", vectorsAll)
    // let min = 100
    // let minFlag = []
    // for(let i=0; i<vectorsAll.length; i++){
    //     for(let j=i+1; j<vectorsAll.length; j++){
    //         let dis = tf.sqrt(tf.sum(tf.squaredDifference(tf.tensor(vectorsAll[i].slice(107,vectorsAll[i].length)),tf.tensor(vectorsAll[j].slice(107,vectorsAll[j].length))))).arraySync()
    //         minFlag = dis < min ? [i, j] : minFlag
    //         min = dis < min ? dis : min
    //     }
    // }
    // console.log(min, minFlag)

    for(let k=1; k<30; k++){
        let vectors = trainVectors
        let pVectors = predictVectors
        let ans = 0
        let convectorsAvg = []
        for(let j=0; j<=k; j++){
            // convectors = EigenvalueCon(vectors, convolutionSize) //利用特征值降维
            convectors = con(vectors, convolutionSize) //利用平均降维
            vectors = convectors[0]
            // console.log("vv", vectors[0][0])
            convectorsAvg = convectors[1]
            //层次聚类
            // const tree = agnes(convectorsAll, {
            //     method: 'ward',
            //     isDistanceMatrix: 'false'
            // });
            // let str = JSON.stringify(tree,"","\t")
            // // fs.writeFileSync('./public/randomForestJson/data.json',str)
            // console.log(str)

            //optics密度聚类
            // let optics = new clustering.OPTICS();
            // // parameters: 2 - neighborhood radius, 2 - number of points in neighborhood to form a cluster
            // let clusters = optics.run(convectorsAll, 0.5, 10);
            // let plot = optics.getReachabilityPlot();
            // console.log("j: ",j,clusters);

            //grid网络聚类
            // console.log(snapList(convectorsAll, 1))

            // //手动选择距离最远的三个点作为三分类的起始聚类中心
            // let center = []
            // let max = 0
            // for(let m=0; m<convectors.length; m++){
            //     for(let n=0; n<convectors.length; n++){
            //         for(let l=0; l<convectors.length; l++){
            //             let dis = tf.sqrt(tf.sum(tf.squaredDifference(tf.tensor(convectors[m]),tf.tensor(convectors[n])))).arraySync()
            //             + tf.sqrt(tf.sum(tf.squaredDifference(tf.tensor(convectors[m]),tf.tensor(convectors[l])))).arraySync()
            //             + tf.sqrt(tf.sum(tf.squaredDifference(tf.tensor(convectors[n]),tf.tensor(convectors[l])))).arraySync()
            //             center = dis>max ? [convectors[m],convectors[n],convectors[l]] : center
            //             max = dis>max ? dis : max
            //         }
            //     }
            // }

            //手动选择距离最远的两个点作为二分类的起始聚类中心
            let center = []
            let max = 0
            for(let m=0; m<convectorsAvg.length; m++){
                for(let n=0; n<convectorsAvg.length; n++){
                    let dis = tf.sqrt(tf.sum(tf.squaredDifference(tf.tensor(convectorsAvg[m]),tf.tensor(convectorsAvg[n])))).arraySync()
                    center = dis>max ? [convectorsAvg[m],convectorsAvg[n]] : center
                    max = dis>max ? dis : max
                }
            }
            //测试平均卷积之后向量类内间距和类间间距的变化
            //结论：当卷积核大小不断加大的时候，类内距离与类间间距不断减小，类内距离的均值与类间距离的均值的比值在趋势上不断加大
            //所以在高卷积的前提下用聚类反而能更容易将类内和类间的向量分辨开
            // let sameSum = tf.scalar(0)
            // let differentSum = tf.scalar(0)
            // let firstClassSum = 3
            // for(let k=0; k<convectors.length; k++){
            //     if(k<firstClassSum){
            //         sameSum = tf.add(tf.sqrt(tf.sum(tf.squaredDifference(tf.tensor(convectors[0]),tf.tensor(convectors[k])))),sameSum)
            //     }else{
            //         differentSum = tf.add(tf.sqrt(tf.sum(tf.squaredDifference(tf.tensor(convectors[0]),tf.tensor(convectors[k])))),differentSum)
            //     }
            // }
            // sameSum = tf.div(sameSum,tf.scalar(firstClassSum-1))
            // differentSum = tf.div(differentSum,tf.scalar(convectors.length-firstClassSum))
            // console.log("value: ")
            // sameSum.print()
            // differentSum.print()
            // console.log("Convolution kernel: ",j," scale: ")
            // tf.div(differentSum,sameSum).print()
            //k-means聚类
            // ans = kmeans(convectorsAvg, 2, { initialization: center, maxIterations:1000})
            ans = kmeans(convectorsAvg, 3, { initialization: 'kmeans++', maxIterations:1000})
            //console.log("time:",j,ans)
        }//for xunhuan

        //训练分类器
        let predictions = []
        for(let i=0; i<vectors.length; i++){
            for(let j=0; j<vectors[i].length; j++){
                predictions.push(ans.clusters[i])
            }
        }
        const options = {
            seed: 3,
            maxFeatures: 0.8,
            replacement: true,
            nEstimators: 100,
            useSampleBagging: true
        };
        let classifer = new RandomForest.RandomForestClassifier(options);
        // console.log(predictions)
        let convectorsAll = []
        for(subVectors of vectors){
            for(val of subVectors){
                convectorsAll.push(val)
            }
        } 
        classifer.train(convectorsAll,predictions) //用所有输入数据及其聚类结果进行分类
        // classifer.train(convectorsAvg,ans.clusters) //用所有输入数据的类间均值及其聚类结果进行分类

        // predict classifer 验证随机森林分类器
        //获得实际分类
        let beforePredictResult = getBeforePredictResult(trainDatapath, predictDatapath, ans.clusters)

        //利用分类器进行分类
        for(let j=0; j<=k; j++){
            //pVectors = EigenvalueCon(pVectors, convolutionSize)[0] //利用特征值降维
            pVectors = con(pVectors, convolutionSize)[0] //利用平均降维
        }
        let predictConVectors = []
        for(subVectors of pVectors){
            for(val of subVectors){
                predictConVectors.push(val)
            }
        } 
        let result = classifer.predict(predictConVectors);
        let accuracy = result.reduce((sum, val, index) =>{ let b = val===beforePredictResult[index] ? 1 : 0; return sum + b}, 0) / result.length
        console.log("classifer accuracy: ", k, "Class1: ", ans.centroids[0].size, "Class2: ", ans.centroids[1].size, accuracy)
    }
}

createTree()