// var fs = require('fs');
const tf = require("@tensorflow/tfjs-node");
// const { agnes } = require('ml-hclust');
// const {Matrix, EigenvalueDecomposition, covariance} = require('ml-matrix');
// const config = require("../configParameter/config.json")

// console.log(config.p)
// let cls_prob = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]]
// //let cls_prob1 = [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]
// //let cls_prob1 = new Array(cls_prob[0].length).fill(new Array(cls_prob.length).fill(99));
// let cls_prob1 = Array(cls_prob[0].length).fill(null).map((val,index1) => {
//     return Array(cls_prob.length).fill(null).map((val,index2)=>{
//         return cls_prob[index2][index1]
//     })
// })

// console.log(cls_prob1)

// let cls_probs = [[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]],[[13,14],[15,16],[17,18]]]

// let cls_probs1 = Array(cls_probs[0][0].length).fill(null).map((val,index0) => {
//     return Array(cls_probs[0].length).fill(null).map((val,index1)=>{
//         return Array(cls_probs.length).fill(null).map((val,index2)=>{
//             return cls_probs[index2][index1][index0]
//         })
//     })
// })

// cls_prob.map((val,index1)=>{
//     val.map((val,index2)=>{
//         console.log(val,index1,index2)
//         console.log(cls_prob1[index2][index1])
//         cls_prob1[index2][index1] = val
//         console.log(cls_prob1)
//     })
// })

// [ 1, 1168, 1280, 3 ]
// let arr = [255]
// let x1 = []
// for(let j=0; j<1280; j++){
//     x1.push([225])  
// }
// let x = []
// for(let j=0; j<1168; j++){
//     x.push([...x1])  
// }
// console.log(x.length, x[0].length, x[0][0].length)

// console.log([...''.padEnd(100)].map((v,i)=>v))

// const input1 = tf.input({shape: [2, 2]});
// const input2 = tf.input({shape: [2, 2]});
// const input3 = tf.input({shape: [2, 2]});
// const multiplyLayer = tf.layers.multiply();
// const product = multiplyLayer.apply([input1, new tf.SymbolicTensor({})]);
// console.log(product.shape);

// let components = []
// const files = fs.readdirSync('./public/images/')
// files.forEach(function (item, index) {
//     let stat = fs.lstatSync("./public/images/" + item)
//     if (stat.isDirectory() === true) { 
//       components.push(item)
//     }
// })
// let res = components.map(item=>{
//     let file = fs.readdirSync('./public/images/'+item+'/')
//     return file.map(val=>{
//         let stat = fs.lstatSync('./public/images/'+item+'/'+val)
//         if(stat.isDirectory() === false){
//             return val
//         }
//     })
// })

// let trainDatapath = './public/images/RandomForestTrainData/'
// let dirArr=[]
// let dir = fs.readdirSync(trainDatapath)
// dir.map(item=>{
//     let stat = fs.lstatSync(trainDatapath + item)
//     if (stat.isDirectory() === true) { 
//         let subDirArr = []
//         let subDir = fs.readdirSync(trainDatapath + item + '/')
//         subDir.map(val=>{
//             let stat = fs.lstatSync(trainDatapath + item + '/' + val)
//             if(stat.isDirectory() === false){
//                 subDirArr.push(trainDatapath + item + '/' + val)
//             }
//         })
//         dirArr.push(subDirArr)
//       }
// })

// console.log(components);
// console.log(dirArr);

// console.log(tf.mean(tf.tensor([[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]), [0,1,2]).arraySync())
// console.log(tf.add(tf.tensor([1,2,3]),tf.mul(tf.tensor([1,2,3]),tf.scalar(0.1))).print())
// console.log(tf.sqrt(tf.sum(tf.squaredDifference(tf.tensor([1,1,1]),tf.tensor([2,2,2])))).arraySync())
// tf.sqrt(tf.sum(tf.squaredDifference(tf.tensor([[1,1,1]]),tf.tensor([[2,2,2]])))).print()
// vectorsAll = [[1,3],[3,4]]
// let tensor = tf.tensor(vectorsAll)
// tensor = tf.sub(tensor, tf.mean(tensor,0))
// tensor = tensor.arraySync()
// console.log(tensor)
// const tree = agnes(convectorsAll, {
//     method: 'complete',
//     isDistanceMatrix: 'false'
// });
// let str = JSON.stringify(tree,"","\t")
// // fs.writeFileSync('./public/randomForestJson/data.json',str)
// console.log(str)
// let vectors = [[[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5]],[[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5]]]
// let real = [0.1,0.2,0.3,0.4,0.5]
// for(let j=2; j<=5; j=j+1){
//     let convolutionSize = j
//     let convectors = []
//     let convectorsAll = []
//     for(subVectors of vectors){
//         let avgVector = tf.scalar(0)
//         for(val of subVectors){
//             let convector = []
//             for(let i=0; i<val.length; i++){
//                 let EigenvalueSum = real.slice(i, i+convolutionSize).reduce(function (a, b) { return a + b;})
//                 console.log("sum:", EigenvalueSum)
//                 let res = i+convolutionSize<=val.length ? val.slice(i, i+convolutionSize).reduce(function (a, b, index) { console.log("real:", index, a, b); return a + b * real[i+index];}, 0)/EigenvalueSum : null
//                 if(res !== null){
//                     convector.push(res)
//                 }    
//             }
            
//             // let out = tf.tensor(val)
//             // let out1 = tf.tensor([...convector,...val.slice(val.length-convolutionSize,val.length)])
//             // let dist = tf.sqrt(tf.sum(tf.squaredDifference(out,out1)))
//             // dist.print()
//             avgVector = tf.add(avgVector, convector)
//             convectorsAll.push(convector)
//         }
//         avgVector = tf.div(avgVector,tf.scalar(subVectors.length))
//         convectors.push(avgVector.arraySync())
//     }
//     console.log("con:",convectors)
//     console.log("all:",convectorsAll)

// let vec = [[[2.5,2.4],[0.5,0.7],[2.2,2.9],[1.9,2.2],[3.1,3.0]],[[2.3,2.7],[2,1.6],[1,1.1],[1.5,1.6],[1.1,0.9]]]
// let mm = [[[-1,-2], [-1,0]], [[5,0], [2,1], [0,1]]]
// function findEigenvalue(vectors){
//     let vectorsAll = []
//     for(subVectors of vectors){
//         for(val of subVectors){
//             vectorsAll.push(val)
//         }
//     } 
//     let tensor = tf.tensor(vectorsAll)
//     tensor = tf.sub(tensor, tf.mean(tensor,0)).arraySync()
//     console.log(tensor)
//     let vectorMatrix = new Matrix(tensor)
//     let covMatrix = covariance(vectorMatrix) //协方差矩阵
//     //console.log(covMatrix)
//     let EigenvalueMatrix = new EigenvalueDecomposition(covMatrix); //特征类
//     let real = EigenvalueMatrix.realEigenvalues //特征数组
//     let EigenvectorMatrix = EigenvalueMatrix.eigenvectorMatrix
//     let sub = new Matrix([EigenvectorMatrix.getRowVector(0).to1DArray()])
//     console.log(vectorMatrix.mmul(sub.transpose()))
//     return [real, EigenvectorMatrix]
// }
// let re = findEigenvalue(vec)
// console.log(re)

// function sortToIndex(originalArr){
//     originalArr = originalArr.map(function(item,index){
//         return {'key':index,'value':parseInt(item)}
//     }) 
//     let indexArr = originalArr.sort(function(a,b){
//         return b.value - a.value
//     }).map(function(item){
//         return item.key
//     })
//     return indexArr
// }

// let [real, EigenvectorMatrix] = findEigenvalue(mm)
// console.log(EigenvectorMatrix)


// function getQuad(z) {
//         let A = z[0], B = z[1], C = z[2];
//         let a = 0, b = 1, c = 2;
//         let n = z.length;
//         while (true) {
//             while (true) {
//                 while (true) {
//                     while (Area(z[a], z[b], z[c]) <= Area(z[a], z[b], z[c], z[(d + 1) % n])) {
//                         d = (d + 1) % n;
//                     }
//                     if (Area(z[a], z[b], z[c], z[d]) > Area(z[a], z[b], z[(c + 1) % n], z[d])) {
//                         break;
//                     }
//                     c = (c + 1) % n;
//                 }
//                 if (Area(z[a], z[b], z[c], z[d]) > Area(z[a], z[(b + 1) % n], z[c], z[d])) {
//                     break;
//                 }
//                 b = (b + 1) % n;
//             }
//             if (Area(z[a], z[b], z[c], z[d]) > Area(A, B, C, D)) {
//                 A = z[a];
//                 B = z[b];
//                 C = z[c];
//                 D = z[d];
//             }
//             a = (a + 1) % n;
//             if (a == b) {
//                 b = (b + 1) % n;
//             }
//             if (b == c) {
//                 c = (c + 1) % n;
//             }
//             if (c == d) {
//                 d = (d + 1) % n;
//             }
//             if (a == 0) {
//                 break;
//             }
//         }

//         return Area(A, B, C, D);
//     }

// var ProgressBar = require('progress');
// var https = require('https');
 
// var req = https.request({
//   host: 'download.github.com',
//   port: 443,
//   path: '/visionmedia-node-jscoverage-0d4608a.zip'
// });
 
// req.on('response', function(res){
//   var len = parseInt(res.headers['content-length'], 10);
 
//   console.log();
//   var bar = new ProgressBar('  downloading [:bar] :rate/bps :percent :etas', {
//     complete: '=',
//     incomplete: ' ',
//     width: 20,
//     total: len
//   });
 
//   res.on('data', function (chunk) {
//     bar.tick(chunk.length);
//   });
 
//   res.on('end', function () {
//     console.log('\n');
//   });
// });
 
// req.end();

// var ProgressBar = require('./progress-bar');
 
// // 初始化一个进度条长度为 50 的 ProgressBar 实例
// var pb = new ProgressBar('下载进度', 50);
 
// // 这里只是一个 pb 的使用示例，不包含任何功能
// var num = 0, total = 200;
// function downloading() {
//  if (num <= total) {
//   // 更新进度条
//   pb.render({ completed: num, total: total });
 
//   num++;
//   setTimeout(function (){
//    downloading();
//   }, 500)
//  }
// }
// downloading();

// var ProgressBar = require('progress');
// var bar = new ProgressBar('  downloading :bar :rate/bps :percent :etas', {
//   complete: '█',
//   incomplete: '░',
//   width: 20,
//   total: 10
// });
// var timer = setInterval(function () {
//   bar.tick();
//   if (bar.complete) {
//     console.log('\ncomplete');
//     clearInterval(timer);
//   }
// }, 500);

var fs = require("fs");

// fs.readFile('./public/appacsv/gt_avg_train.csv', function (err, data) {
//     var table = new Array();
//     if (err) {
//         console.log(err.stack);
//         return;
//     }

//     ConvertToTable(data, function (table) {
//         console.log(table);
//     })
// });

// let data = fs.readFileSync('./public/appacsv/gt_avg_train.csv')
// data = ConvertToTable(data)
// console.log(data);
// console.log("程序执行完毕");

// function ConvertToTable(data) {
//     data = data.toString();
//     var table = new Array();
//     var rows = new Array();
//     // console.log(data)
//     rows = data.split("\n");
//     for (var i = 0; i < rows.length; i++) {
//         table.push(rows[i].split(","));
//     }
//     return table
// }

// const a = tf.tensor([[1, 2]]);
// const b = tf.tensor([[3, 4]]);
// const c = tf.tensor([[5, 6]]);
// tf.concat([a, b, c]).print();
let y = new Array(10).fill(new Array(101).fill(0))
console.log(y)