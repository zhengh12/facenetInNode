const RandomForest = require('ml-random-forest') 
// const IrisDataset = require('ml-dataset-iris') 
const facenet = require("./facenet")
const fs = require("fs");

// var trainingSet = IrisDataset.getNumbers();
// var predictions = IrisDataset.getClasses().map((elem) =>
//   IrisDataset.getDistinctClasses().indexOf(elem)
// );

// let trainingSets = trainingSet.map((val,index)=>{
//     return trainingSet[50 * (index % 3) + Math.floor(index / 3)]
// })

// let trainingSet0 = trainingSet.map((val,index)=>{
//     return trainingSet[50 * (index % 3) + Math.floor(index / 3)].slice(0,3)
// })

// let trainingSet1 = trainingSet.map((val,index)=>{
//     return trainingSet[50 * (index % 3) + Math.floor(index / 3)].slice(1,4)
// })

// let trainingSet2 = trainingSet.map((val,index)=>{
//     let res = trainingSet[50 * (index % 3) + Math.floor(index / 3)]
//     return [res[0],res[1],res[3]]
// })

// let trainingSet3 = trainingSet.map((val,index)=>{
//     let res = trainingSet[50 * (index % 3) + Math.floor(index / 3)]
//     return [res[0],res[2],res[3]]
// })

// predictions = predictions.map((val,index)=>{
//     return predictions[50 * (index % 3) + Math.floor(index / 3)]
// })

// console.log("trainingSet0",trainingSet0)
// console.log("trainingSet1",trainingSet1)
// console.log("trainingSet2",trainingSet2)
// console.log("trainingSet3",trainingSet3)
// console.log(predictions)
// let begin1 = 0
// let begin2 = 120
// let end = 150
// let trainingSets0 = trainingSets.slice(begin1,begin2)
// let trainingSets1 = trainingSets.slice(begin2,end)
// let trainingSet00 = trainingSet0.slice(begin1,begin2)
// let trainingSet01 = trainingSet0.slice(begin2,end)
// let trainingSet10 = trainingSet1.slice(begin1,begin2)
// let trainingSet11 = trainingSet1.slice(begin2,end)
// let trainingSet20 = trainingSet2.slice(begin1,begin2)
// let trainingSet21 = trainingSet2.slice(begin2,end)
// let trainingSet30 = trainingSet3.slice(begin1,begin2)
// let trainingSet31 = trainingSet3.slice(begin2,end)
// let predictions0 = predictions.slice(begin1,begin2)

// var options = {
//     seed: 3,
//     maxFeatures: 0.8,
//     replacement: true,
//     nEstimators: 25
//   };

// var classifiers = new RandomForest.RandomForestClassifier(options);
// classifiers.train(trainingSets0, predictions0);

// var classifier0 = new RandomForest.RandomForestClassifier(options);
// classifier0.train(trainingSet00, predictions0);

// var classifier1 = new RandomForest.RandomForestClassifier(options);
// classifier1.train(trainingSet10, predictions0);

// var classifier2 = new RandomForest.RandomForestClassifier(options);
// classifier2.train(trainingSet20, predictions0);

// var classifier3 = new RandomForest.RandomForestClassifier(options);
// classifier3.train(trainingSet30, predictions0);

// // console.log(classifier.toJSON())
// var results = classifiers.predict(trainingSets1);
// var result0 = classifier0.predict(trainingSet01);
// var result1 = classifier1.predict(trainingSet11);
// var result2 = classifier2.predict(trainingSet21);
// var result3 = classifier3.predict(trainingSet31);
// console.log("results",results)
// result0.map((val,index)=>{
//     console.log("result0",index,result0[index])
//     console.log("result1",index,result1[index])
//     console.log("result2",index,result2[index])
//     console.log("result3",index,result3[index])
// })

async function trainRandomForest() {

    // const imagePath0 = "./public/images/LarryPage/Larry_Page_0000.jpg"
    // const imagePath1 = "./public/images/LarryPage/Larry_Page_0001.jpg"
    // const imagePath2 = "./public/images/LarryPage/Larry_Page_0002.jpg"
    // const imagePath3 = "./public/images/BillGates/Bill_Gates_0000.jpg"
    // const imagePath4 = "./public/images/BillGates/Bill_Gates_0001.jpg"
    // const imagePath5 = "./public/images/BillGates/Bill_Gates_0002.jpg"
    // const imagePath6 = "./public/images/MarkZuckerberg/Mark_Zuckerberg_0000.jpg"
    // const imagePath7 = "./public/images/MarkZuckerberg/Mark_Zuckerberg_0001.jpg"
    // const imagePath8 = "./public/images/MarkZuckerberg/Mark_Zuckerberg_0002.jpg"
    // const imagePath9 = "./public/images/TaylorSwift/TaylorSwift0001.jpg"
    // const imagePath10 = "./public/images/TaylorSwift/TaylorSwift0002.jpg"
    // const imagePath11 = "./public/images/TaylorSwift/TaylorSwift0003.jpeg"
    // const modelPath = "./public/model/Facenet1/model.json"
    // let vector0 = await facenet.faceVector(modelPath,imagePath0)
    // vector0 = vector0.arraySync()[0]
    // vector0 = vector0.map((val,index)=>{
    //     return vector0.slice(0,index).concat(vector0.slice(index+1,vector0.length))
    // })
    // let vector1 = await facenet.faceVector(modelPath,imagePath1)
    // vector1 = vector1.arraySync()[0]
    // vector1 = vector1.map((val,index)=>{
    //     return vector1.slice(0,index).concat(vector1.slice(index+1,vector1.length))
    // })
    // let vector2 = await facenet.faceVector(modelPath,imagePath2)
    // vector2 = vector2.arraySync()[0]
    // vector2 = vector2.map((val,index)=>{
    //     return vector2.slice(0,index).concat(vector2.slice(index+1,vector2.length))
    // })
    // let vector3 = await facenet.faceVector(modelPath,imagePath3)
    // vector3 = vector3.arraySync()[0]
    // vector3 = vector3.map((val,index)=>{
    //     return vector3.slice(0,index).concat(vector3.slice(index+1,vector3.length))
    // })
    // let vector4 = await facenet.faceVector(modelPath,imagePath4)
    // vector4 = vector4.arraySync()[0]
    // vector4 = vector4.map((val,index)=>{
    //     return vector4.slice(0,index).concat(vector4.slice(index+1,vector4.length))
    // })
    // let vector5 = await facenet.faceVector(modelPath,imagePath5)
    // vector5 = vector5.arraySync()[0]
    // vector5 = vector5.map((val,index)=>{
    //     return vector5.slice(0,index).concat(vector5.slice(index+1,vector5.length))
    // })
    // let vector6 = await facenet.faceVector(modelPath,imagePath6)
    // vector6 = vector6.arraySync()[0]
    // vector6 = vector6.map((val,index)=>{
    //     return vector6.slice(0,index).concat(vector6.slice(index+1,vector6.length))
    // })
    // let vector7 = await facenet.faceVector(modelPath,imagePath7)
    // vector7 = vector7.arraySync()[0]
    // vector7 = vector7.map((val,index)=>{
    //     return vector7.slice(0,index).concat(vector7.slice(index+1,vector7.length))
    // })
    // let vector8 = await facenet.faceVector(modelPath,imagePath8)
    // vector8 = vector8.arraySync()[0]
    // vector8 = vector8.map((val,index)=>{
    //     return vector8.slice(0,index).concat(vector8.slice(index+1,vector8.length))
    // })
    // let vector9 = await facenet.faceVector(modelPath,imagePath9)
    // vector9 = vector9.arraySync()[0]
    // vector9 = vector9.map((val,index)=>{
    //     return vector9.slice(0,index).concat(vector9.slice(index+1,vector9.length))
    // })
    // let vector10 = await facenet.faceVector(modelPath,imagePath10)
    // vector10 = vector10.arraySync()[0]
    // vector10 = vector10.map((val,index)=>{
    //     return vector10.slice(0,index).concat(vector10.slice(index+1,vector10.length))
    // })
    // let vector11 = await facenet.faceVector(modelPath,imagePath11)
    // vector11 = vector11.arraySync()[0]
    // vector11 = vector11.map((val,index)=>{
    //     return vector11.slice(0,index).concat(vector11.slice(index+1,vector11.length))
    // })
    //TrainData = [vector0.arraySync()[0],vector1.arraySync()[0],vector3.arraySync()[0],vector4.arraySync()[0],vector6.arraySync()[0],vector7.arraySync()[0]]
    let trainDatapath = './public/images/RandomForestTrainData/'
    let dirArr=[]
    let dir = fs.readdirSync(trainDatapath)
    dir.map(item=>{
        let stat = fs.lstatSync(trainDatapath + item)
        if (stat.isDirectory() === true) { 
            let subDirArr = []
            let subDir = fs.readdirSync(trainDatapath + item + '/')
            subDir.map(val=>{
                let stat = fs.lstatSync(trainDatapath + item + '/' + val)
                if(stat.isDirectory() === false){
                    subDirArr.push(trainDatapath + item + '/' + val)
                }
            })
            dirArr.push(subDirArr)
        }
    })
    
    console.log(dirArr)
    const modelPath = "./public/model/Facenet1/model.json"
    let vectors = []
    dirArr.map(async(val,indexs)=>{
        await val.map(async val=>{
            let vector = await facenet.faceVector(modelPath,val)
            vector = vector.arraySync()[0]
            vector = vector.map((val,index)=>{
                return vector.slice(0,index).concat(vector.slice(index+1,vector.length)).push(index)
            })
            vectors.push(vector)
        })
    })
    console.log(vectors)
    babalala
    let predictions = [0,0,1,1,2,2,3,3]
    var options = {
        seed: 3,
        maxFeatures: 0.8,
        replacement: true,
        nEstimators: 25
    };
    let classifier = vector0.map(val=>{
        return new RandomForest.RandomForestClassifier(options);
    })
    classifier.map((val,index)=>{
        val.train([vector0[index],vector1[index],vector3[index],vector4[index],vector6[index],vector7[index],vector9[index],vector10[index]],predictions)
    })
    let str = classifier.map(val=>{
        return val.toJSON()
    })
    console.log(str)
    str = JSON.stringify(str,"","\t")

    fs.writeFileSync('./public/randomForestJson/data.json',str)

    classifier.map((val,index)=>{
        let results = val.predict([vector1[index],vector5[index],vector8[index],vector11[index]]);
        console.log(results)
    })
}

async function predictFace(){
    let str = fs.readFileSync('./public/randomForestJson/data.json', 'utf-8')
    str = JSON.parse(str)
    var options = {
        seed: 3,
        maxFeatures: 0.8,
        replacement: true,
        nEstimators: 25
    };
    let classifier = str.map((val,index)=>{
        let cla = new RandomForest.RandomForestClassifier(true,val);
        // cla.indexes = val.baseModel.indexes
        // cla.n = val.baseModel.n
        // cla.isClassifier = val.baseModel.isClassifier
        // cla.estimators = val.baseModel.estimators
        // cla.useSampleBagging = val.baseModel.useSampleBagging
        return cla
    })
    console.log(classifier)
    const modelPath = "./public/model/Facenet1/model.json"
    const imagePath4 = "./public/images/BillGates/Bill_Gates_0001.jpg"
    const imagePath5 = "./public/images/BillGates/Bill_Gates_0002.jpg"
    const imagePath2 = "./public/images/LarryPage/Larry_Page_0002.jpg"
    const imagePath8 = "./public/images/MarkZuckerberg/Mark_Zuckerberg_0002.jpg"
    const imagePath11 = "./public/images/TaylorSwift/TaylorSwift0003.jpeg"
    let vector4 = await facenet.faceVector(modelPath,imagePath4)
    vector4 = vector4.arraySync()[0]
    vector4 = vector4.map((val,index)=>{
        return vector4.slice(0,index).concat(vector4.slice(index+1,vector4.length))
    })

    let vector5 = await facenet.faceVector(modelPath,imagePath5)
    vector5 = vector5.arraySync()[0]
    vector5 = vector5.map((val,index)=>{
        return vector5.slice(0,index).concat(vector5.slice(index+1,vector5.length))
    })
    let vector11 = await facenet.faceVector(modelPath,imagePath11)
    vector11 = vector11.arraySync()[0]
    vector11 = vector11.map((val,index)=>{
        return vector11.slice(0,index).concat(vector11.slice(index+1,vector11.length))
    })
    let vector8 = await facenet.faceVector(modelPath,imagePath8)
    vector8 = vector8.arraySync()[0]
    vector8 = vector8.map((val,index)=>{
        return vector8.slice(0,index).concat(vector8.slice(index+1,vector8.length))
    })
    let vector2 = await facenet.faceVector(modelPath,imagePath2)
    vector2 = vector2.arraySync()[0]
    vector2 = vector2.map((val,index)=>{
        return vector2.slice(0,index).concat(vector2.slice(index+1,vector2.length))
    })
    // classifier.map((val,index)=>{
    //     val.train([vector4[index]],[1]);
    // })
    console.log(classifier[0].estimators[0])
    classifier.map((val,index)=>{
        let results = val.predict([vector2[index],vector5[index],vector8[index],vector11[index]]);
        console.log(results)
    })
}
// predictFace()
trainRandomForest()
// function randomsort(arr){
//     return arr.sort((a,b)=>{
//         return Math.random()>.5 ? -1 : 1
//     })
// }