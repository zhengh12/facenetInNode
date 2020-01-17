const RandomForest = require('ml-random-forest') 
// const IrisDataset = require('ml-dataset-iris') 
const facenet = require("./facenet")
const fs = require("fs");

//该函数是训练随机森林分类器的
async function trainRandomForest() {
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