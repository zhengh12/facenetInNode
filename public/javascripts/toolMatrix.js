const config = require("../configParameter/config.json")

//根据输入图像的宽高生成图像金字塔比例数组
//input parameters: 
//  Number h 原图像的高
//  Number w 原图像的宽
//output:
//  Number array[n] scales 表示缩放比例的数组
function calculateScales(h, w){
    let pr_scale = 1.0
    if (Math.min(w,h)>500){
        pr_scale = 500.0/Math.min(h,w)
        w = w*pr_scale
        h = h*pr_scale
    }
    else if (Math.max(w,h)<500){
        pr_scale = 500.0/Math.max(h,w)
        w = w*pr_scale
        h = h*pr_scale
    }
    //multi-scale
    let scales = []
    let factor = config.mtcnnParam.imagePyramidFactor //表示缩小后面积是原面积的2/3，factor=根号2/3，值约接近1则人脸检测准度越高，但同样时间花费也高
    let factor_count = 0
    let minl = Math.min(h,w)
    while (minl >= config.mtcnnParam.imagePyramidMin){ //表示图像金字塔最小的像素大小
        scales.push(pr_scale*Math.pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    }
    return scales
}


//非极大值抑制算法
//input parameters: 
//  array[][] rectangles 所有的人脸矩形预测框
//  float threshold 表示废弃候选框需要达到的重合度
//  String type 该算法有“iom”和“iou”两种方式
//output:
//  array[][] result_rectangle 经过处理后的人脸矩形预测框
function NMS(rectangles,threshold,type){
    if(rectangles.length==0){
        return rectangles
    }
    let boxes = rectangles
    //x1 = boxes[:,0]
    let x1 = boxes.map(val=>{return val[0]})
    let y1 = boxes.map(val=>{return val[1]})
    let x2 = boxes.map(val=>{return val[2]})
    let y2 = boxes.map(val=>{return val[3]})
    let s  = boxes.map(val=>{return val[4]})
    //area = np.multiply(x2-x1+1, y2-y1+1)
    let area = x1.map((val,index)=>{
        return (x2[index]-val+1) * (y2[index]-y1[index]+1)
    })
    //console.log(area)
    // I = np.array(s.argsort())
    let I1 = s
    I1.sort(function(a,b){
        return a - b;
    })
    s  = boxes.map(val=>{return val[4]})
    let I = []
    I1.map(val=>{
        s.map((vals,index)=>{
            if(val === vals){
                I.push(index)
            } 
        })
    })
    // console.log(I)
    let pick = []
    // while len(I)>0:
    //     xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) //I[-1] have hightest prob score, I[0:-1]->others
    //     yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
    //     xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
    //     yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
    //     w = np.maximum(0.0, xx2 - xx1 + 1)
    //     h = np.maximum(0.0, yy2 - yy1 + 1)
    //     inter = w * h
    //     if type == 'iom':
    //         o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
    //     else:
    //         o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
    //     pick.append(I[-1])
    //     I = I[np.where(o<=threshold)[0]]
    while(I.length>0){
        let II = I[I.length-1]
        I.splice(I.length-1,1)
        // let x11 = x1[I[I.length-1]]
        // let y11 = y1[I[I.length-1]]
        // let x22 = x2[I[I.length-1]]
        // let y22 = y2[I[I.length-1]]
        // x1.splice(I[I.length-1],1)
        // y1.splice(I[I.length-1],1)
        // x2.splice(I[I.length-1],1)
        // y2.splice(I[I.length-1],1)
        let xx1 = I.map(val=>{
            return Math.max(x1[val],x1[II])
        })
        let yy1 = I.map(val=>{
            return Math.max(y1[val],y1[II])
        })
        let xx2 = I.map(val=>{
            return Math.min(x2[val],x2[II])
        })
        let yy2 = I.map(val=>{
            return Math.min(y2[val],y2[II])
        })
        // console.log(xx1)
        // console.log(yy1)
        // console.log(xx2)
        // console.log(yy2)
        //Math.max(0.0, xx2 - xx1 + 1)
        let w = xx1.map((val,index)=>{
            return Math.max(0.0, xx2[index] - val + 1)
        })
        // let h = Math.max(0.0, yy2 - yy1 + 1)
        let h = yy1.map((val,index)=>{
            return Math.max(0.0, yy2[index] - val + 1)
        })
        //let inter = w * h
        let inter = w.map((val,index)=>{
            return val * h[index]
        })
        // console.log("inter",inter)
        // console.log("area:",area)
        // console.log(type)
        // console.log(type === 'iom')
        let o = type === 'iom' ?  
        I.map((val,index)=>{
            // console.log("iom")
            return inter[index] / (area[II]<=area[val] ? area[II] : area[val]) 
        }) :
        I.map((val,index)=>{
            // console.log("iou")
            return inter[index] / (area[II] + area[val] - inter[index])
        })
        // console.log("o",o)
        pick.push(II)
        // console.log("pick",pick)
        //I = I[np.where(o<=threshold)[0]]
        let III=[]
        o.map((val,index)=>{
            if(val<=threshold){
                III.push(index)
            }
        })
        I.push(II)
        //console.log(I)
        I = III.map(val=>{
            return I[val]
        })
        // console.log(I)
    }
    // result_rectangle = boxes[pick].tolist()
    let result_rectangle = pick.map(val=>{
        return boxes[val]
    })
    // console.log("result_rectangle:",result_rectangle)
    return result_rectangle
}


//将矩形框重构成方形框
//input parameters: 
//  array[][] rectangles 若干人脸矩形预测框
//output:
//  array[][] rectangles 若干人脸方形预测框
function rect2square(rectangles){
    //w = rectangles[:,2] - rectangles[:,0]
    let w = rectangles.map(val=>{
        return val[2]-val[0]
    })
    //h = rectangles[:,3] - rectangles[:,1]
    let h = rectangles.map(val=>{
        return val[3]-val[1]
    })
    //l = np.maximum(w,h).T
    let l = w.map((val,index)=>{
        return val>h[index] ? val : h[index] //两个之中取最大值
    })
    // console.log("l:",l)
    //rectangles[:,0] = rectangles[:,0] + w*0.5 - l*0.5
    // rectangles[:,1] = rectangles[:,1] + h*0.5 - l*0.5 
    // rectangles[:,2:4] = rectangles[:,0:2] + np.repeat([l], 2, axis = 0).T 
    rectangles = rectangles.map((val,index)=>{
        let res = []
        res.push(val[0] + w[index]*0.5 - l[index]*0.5)
        res.push(val[1] + h[index]*0.5 - l[index]*0.5)
        res.push(val[0] + w[index]*0.5 - l[index]*0.5 + l[index])
        res.push(val[1] + h[index]*0.5 - l[index]*0.5 + l[index])
        res.push(val[4])
        return res
    })
    return rectangles
}

function detect_face_12net(cls_prob,roi,out_side,scale,width,height,threshold){
    let in_side = 2*out_side+11
    let stride = 0
    if(out_side != 1){
        stride = (in_side-12)/(out_side-1)
    }
    //(x,y) = np.where(cls_prob>=threshold)
    //////////////////////////////
    let x = []
    let y = []
    cls_prob.map((val,index0)=>{
        return val.map((val,index1)=>{
            if(val>=threshold){
                x.push(index0)
                y.push(index1)
            }
        }) 
    })

    let boundingbox = [x,y]
    boundingbox = Array(boundingbox[0].length).fill(null).map((val,index1) => {
        return Array(boundingbox.length).fill(null).map((val,index2)=>{
            return boundingbox[index2][index1]
        })
    })
    // boundingbox = np.array([x,y]).T
    // bb1 = np.fix((stride * (boundingbox) + 0 ) * scale)
    let bb1 = boundingbox.map(val=>{
        return val.map(val=>{
            return Math.floor((val * stride + 0) * scale/1.0)
        })
    })
    // bb2 = np.fix((stride * (boundingbox) + 11) * scale)
    let bb2 = boundingbox.map(val=>{
        return val.map(val=>{
            return Math.floor((val * stride + 11) * scale/1.0)
        })
    })
    //console.log(bb2,bb2.length)
    // boundingbox = np.concatenate((bb1,bb2),axis = 1)
    let arr1 = []
    bb1.map((val,index0)=>{
        let arr2 = []
        val.map((val,index1)=>{
            arr2.splice(index1,0,val) //保证是以x1,y1,x2,y2顺序
            arr2.push(bb2[index0][index1])
        })
        arr1.push(arr2)
    })
    boundingbox = arr1
    //console.log(boundingbox,boundingbox.length)
    // dx1 = roi[0][x,y]
    let dx1 = x.map((val,index)=>{
        return roi[0][val][y[index]]
    })
    //console.log("cls_prob",cls_prob)
    //console.log(x,y)
    //console.log(dx1,dx1.length)
    // dx2 = roi[1][x,y]
    let dx2 = x.map((val,index)=>{
        return roi[1][val][y[index]]
    })
    // dx3 = roi[2][x,y]
    let dx3 = x.map((val,index)=>{
        return roi[2][val][y[index]]
    })
    // dx4 = roi[3][x,y]
    let dx4 = x.map((val,index)=>{
        return roi[3][val][y[index]]
    })
    // score = np.array([cls_prob[x,y]]).T
    let score = x.map((val,index)=>{
        return [cls_prob[val][y[index]]]
    })
    //console.log(score)
    // offset = np.array([dx1,dx2,dx3,dx4]).T
    // console.log("dx1:",dx1)
    // console.log("dx2:",dx2)
    // console.log("dx3:",dx3)
    // console.log("dx4:",dx4)
    let offset = dx1.map((val,index)=>{
        return [val,dx2[index],dx3[index],dx4[index]]
    })
    //console.log(offset)
    // boundingbox = boundingbox + offset*12.0*scale
    boundingbox = boundingbox.map((val,index0)=>{
        return val.map((val,index1)=>{
            return val + offset[index0][index1]*12.0*scale
        })
    })
    //console.log("boundingbox",boundingbox)
    // rectangles = np.concatenate((boundingbox,score),axis=1)
    let rectangles = boundingbox.map((val,index)=>{
        return val.concat(score[index])
    })
    //console.log("rectangles:",rectangles)
    rectangles = rect2square(rectangles)
    // let rectangless = rect2square([[1,4,2,6,0.3],[1,4,2,7,0.2],[1,5,2,6,0.1]]) //测试数据
    let pick = []
    // for i in range(len(rectangles)){
    //     x1 = int(max(0     ,rectangles[i][0]))
    //     y1 = int(max(0     ,rectangles[i][1]))
    //     x2 = int(min(width ,rectangles[i][2]))
    //     y2 = int(min(height,rectangles[i][3]))
    //     sc = rectangles[i][4]
    //     if x2>x1 and y2>y1:
    //         pick.append([x1,y1,x2,y2,sc])
    // }
    rectangles.map(val=>{
        let x1 = Math.floor(Math.max(0,val[0]))
        let y1 = Math.floor(Math.max(0,val[1]))
        let x2 = Math.floor(Math.min(width,val[2]))
        let y2 = Math.floor(Math.min(height,val[3]))
        let sc = val[4]
        if(x2>x1 && y2>y1){
            pick.push([x1,y1,x2,y2,sc])
        }
    })
    return NMS(pick,0.3,'iou')
}

function filter_face_24net(cls_prob,roi,rectangles,width,height,threshold){
    //prob = cls_prob[:,1]
    let prob = cls_prob.map(val=>{
        return val[1]
    })
    // pick = np.where(prob>=threshold)
    let pick = []
    prob.map((val,index)=>{
        if(val>=threshold){
            pick.push(index)
        }
    })
    //rectangles = np.array(rectangles)
    // x1  = rectangles[pick,0]
    // y1  = rectangles[pick,1]
    // x2  = rectangles[pick,2]
    // y2  = rectangles[pick,3]
    // sc  = np.array([prob[pick]]).T
    let x1 = [pick.map(val=>{
        return rectangles[val][0]
    })]
    let y1 = [pick.map(val=>{
        return rectangles[val][1]
    })]
    let x2 = [pick.map(val=>{
        return rectangles[val][2]
    })]
    let y2 = [pick.map(val=>{
        return rectangles[val][3]
    })]
    let sc = pick.map(val=>{
        return [prob[val]]
    })
    // console.log(sc,sc.length)
    // dx1 = roi[pick,0]
    // dx2 = roi[pick,1]
    // dx3 = roi[pick,2]
    // dx4 = roi[pick,3]
    let dx1 = [pick.map(val=>{
        return roi[val][0]
    })]
    let dx2 = [pick.map(val=>{
        return roi[val][1]
    })]
    let dx3 = [pick.map(val=>{
        return roi[val][2]
    })]
    let dx4 = [pick.map(val=>{
        return roi[val][3]
    })]
    // w   = x2-x1
    // h   = y2-y1
    let w = [x1[0].map((val,index)=>{
        return x2[0][index]-val
    })]
    let h = [y1[0].map((val,index)=>{
        return y2[0][index]-val
    })]
    // console.log(w,w.length)
    // x1  = np.array([(x1+dx1*w)[0]]).T
    // y1  = np.array([(y1+dx2*h)[0]]).T
    // x2  = np.array([(x2+dx3*w)[0]]).T
    // y2  = np.array([(y2+dx4*h)[0]]).T
    x1 = x1[0].map((val,index)=>{
        return [val+dx1[0][index]*w[0][index]]
    })
    y1 = y1[0].map((val,index)=>{
        return [val+dx2[0][index]*h[0][index]]
    })
    x2 = x2[0].map((val,index)=>{
        return [val+dx3[0][index]*w[0][index]]
    })
    y2 = y2[0].map((val,index)=>{
        return [val+dx4[0][index]*h[0][index]]
    })
    //rectangles = np.concatenate((x1,y1,x2,y2,sc),axis=1)
    rectangles = x1.map((val,index)=>{
        return val.concat(y1[index]).concat(x2[index]).concat(y2[index]).concat(sc[index])
    })
    rectangles = rect2square(rectangles)
    pick = []
    // for i in range(len(rectangles)):
    //     x1 = int(max(0     ,rectangles[i][0]))
    //     y1 = int(max(0     ,rectangles[i][1]))
    //     x2 = int(min(width ,rectangles[i][2]))
    //     y2 = int(min(height,rectangles[i][3]))
    //     sc = rectangles[i][4]
    //     if x2>x1 and y2>y1:
    //         pick.append([x1,y1,x2,y2,sc])
    rectangles.map(val=>{
        let x1 = Math.floor(Math.max(0,val[0]))
        let y1 = Math.floor(Math.max(0,val[1]))
        let x2 = Math.floor(Math.min(width,val[2]))
        let y2 = Math.floor(Math.min(height,val[3]))
        let sc = val[4]
        if(x2>x1 && y2>y1){
            pick.push([x1,y1,x2,y2,sc])
        }
    })
    return NMS(pick,0.3,'iou')
}

function filter_face_48net(cls_prob,roi,pts,rectangles,width,height,threshold){
    //prob = cls_prob[:,1]
    let prob = cls_prob.map(val=>{
        return val[1]
    })
    //pick = np.where(prob>=threshold)
    let pick = []
    prob.map((val,index)=>{
        if(val>=threshold){
            pick.push(index)
        }
    })
    //rectangles = np.array(rectangles)
    // x1  = rectangles[pick,0]
    // y1  = rectangles[pick,1]
    // x2  = rectangles[pick,2]
    // y2  = rectangles[pick,3]
    // sc  = np.array([prob[pick]]).T
    let x1 = [pick.map(val=>{
        return rectangles[val][0]
    })]
    let y1 = [pick.map(val=>{
        return rectangles[val][1]
    })]
    let x2 = [pick.map(val=>{
        return rectangles[val][2]
    })]
    let y2 = [pick.map(val=>{
        return rectangles[val][3]
    })]
    let sc = pick.map(val=>{
        return [prob[val]]
    })
    // dx1 = roi[pick,0]
    // dx2 = roi[pick,1]
    // dx3 = roi[pick,2]
    // dx4 = roi[pick,3]
    let dx1 = [pick.map(val=>{
        return roi[val][0]
    })]
    let dx2 = [pick.map(val=>{
        return roi[val][1]
    })]
    let dx3 = [pick.map(val=>{
        return roi[val][2]
    })]
    let dx4 = [pick.map(val=>{
        return roi[val][3]
    })]
    // w   = x2-x1
    // h   = y2-y1
    let w = [x1[0].map((val,index)=>{
        return x2[0][index]-val
    })]
    let h = [y1[0].map((val,index)=>{
        return y2[0][index]-val
    })]
    // pts0= np.array([(w*pts[pick,0]+x1)[0]]).T
    // pts1= np.array([(h*pts[pick,5]+y1)[0]]).T
    // pts2= np.array([(w*pts[pick,1]+x1)[0]]).T
    // pts3= np.array([(h*pts[pick,6]+y1)[0]]).T
    // pts4= np.array([(w*pts[pick,2]+x1)[0]]).T
    // pts5= np.array([(h*pts[pick,7]+y1)[0]]).T
    // pts6= np.array([(w*pts[pick,3]+x1)[0]]).T
    // pts7= np.array([(h*pts[pick,8]+y1)[0]]).T
    // pts8= np.array([(w*pts[pick,4]+x1)[0]]).T
    // pts9= np.array([(h*pts[pick,9]+y1)[0]]).T
    let pts0 = pick.map((val,index)=>{
        return [pts[val][0]*w[0][index] + x1[0][index]]
    })
    let pts1 = pick.map((val,index)=>{
        return [pts[val][5]*h[0][index] + y1[0][index]]
    })
    let pts2 = pick.map((val,index)=>{
        return [pts[val][1]*w[0][index] + x1[0][index]]
    })
    let pts3 = pick.map((val,index)=>{
        return [pts[val][6]*h[0][index] + y1[0][index]]
    })
    let pts4 = pick.map((val,index)=>{
        return [pts[val][2]*w[0][index] + x1[0][index]]
    })
    let pts5 = pick.map((val,index)=>{
        return [pts[val][7]*h[0][index] + y1[0][index]]
    })
    let pts6 = pick.map((val,index)=>{
        return [pts[val][3]*w[0][index] + x1[0][index]]
    })
    let pts7 = pick.map((val,index)=>{
        return [pts[val][8]*h[0][index] + y1[0][index]]
    })
    let pts8 = pick.map((val,index)=>{
        return [pts[val][4]*w[0][index] + x1[0][index]]
    })
    let pts9 = pick.map((val,index)=>{
        return [pts[val][9]*h[0][index] + y1[0][index]]
    })
    // # pts0 = np.array([(w * pts[pick, 0] + x1)[0]]).T
    // # pts1 = np.array([(h * pts[pick, 1] + y1)[0]]).T
    // # pts2 = np.array([(w * pts[pick, 2] + x1)[0]]).T
    // # pts3 = np.array([(h * pts[pick, 3] + y1)[0]]).T
    // # pts4 = np.array([(w * pts[pick, 4] + x1)[0]]).T
    // # pts5 = np.array([(h * pts[pick, 5] + y1)[0]]).T
    // # pts6 = np.array([(w * pts[pick, 6] + x1)[0]]).T
    // # pts7 = np.array([(h * pts[pick, 7] + y1)[0]]).T
    // # pts8 = np.array([(w * pts[pick, 8] + x1)[0]]).T
    // # pts9 = np.array([(h * pts[pick, 9] + y1)[0]]).T
    // x1  = np.array([(x1+dx1*w)[0]]).T
    // y1  = np.array([(y1+dx2*h)[0]]).T
    // x2  = np.array([(x2+dx3*w)[0]]).T
    // y2  = np.array([(y2+dx4*h)[0]]).T
    x1 = x1[0].map((val,index)=>{
        return [val+dx1[0][index]*w[0][index]]
    })
    y1 = y1[0].map((val,index)=>{
        return [val+dx2[0][index]*h[0][index]]
    })
    x2 = x2[0].map((val,index)=>{
        return [val+dx3[0][index]*w[0][index]]
    })
    y2 = y2[0].map((val,index)=>{
        return [val+dx4[0][index]*h[0][index]]
    })
    //rectangles=np.concatenate((x1,y1,x2,y2,sc,pts0,pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9),axis=1)
    rectangles = x1.map((val,index)=>{
        return val.concat(y1[index]).concat(x2[index]).concat(y2[index]).concat(sc[index]).concat(pts0[index]).concat(pts1[index])
        .concat(pts2[index]).concat(pts3[index]).concat(pts4[index]).concat(pts5[index]).concat(pts6[index]).concat(pts7[index])
        .concat(pts8[index]).concat(pts9[index])
    })
    pick = []
    // for i in range(len(rectangles)):
    //     x1 = int(max(0     ,rectangles[i][0]))
    //     y1 = int(max(0     ,rectangles[i][1]))
    //     x2 = int(min(width ,rectangles[i][2]))
    //     y2 = int(min(height,rectangles[i][3]))
    //     if x2>x1 and y2>y1:
    //         pick.append([x1,y1,x2,y2,rectangles[i][4],
    //              rectangles[i][5],rectangles[i][6],rectangles[i][7],rectangles[i][8],rectangles[i][9],rectangles[i][10],rectangles[i][11],rectangles[i][12],rectangles[i][13],rectangles[i][14]])
    rectangles.map((val,i)=>{
        let x1 = Math.floor(Math.max(0,val[0]))
        let y1 = Math.floor(Math.max(0,val[1]))
        let x2 = Math.floor(Math.min(width,val[2]))
        let y2 = Math.floor(Math.min(height,val[3]))
        if(x2>x1 && y2>y1){
            pick.push([x1,y1,x2,y2,rectangles[i][4],rectangles[i][5],rectangles[i][6],rectangles[i][7],
                rectangles[i][8],rectangles[i][9],rectangles[i][10],rectangles[i][11],rectangles[i][12],rectangles[i][13],
                rectangles[i][14]])
        }
    })
    // console.log("pppiiiccckkk:",pick)
    return NMS(pick,0.3,'iom')
}

exports.calculateScales = calculateScales
exports.detect_face_12net = detect_face_12net
exports.NMS = NMS
exports.filter_face_24net = filter_face_24net
exports.filter_face_48net = filter_face_48net