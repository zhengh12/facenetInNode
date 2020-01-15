const tf = require("@tensorflow/tfjs-node");

function inception_resnet_stem(input){
    // if K.image_dim_ordering() == "th":
    //     channel_axis = 1
    // else:
    //     channel_axis = -1
    let channel_axis = true ? -1 : 1
    //# Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    // c = Convolution2D(32, 3, 3, activation='relu', subsample=(2, 2))(input)
    let c = tf.layers.conv2d({filters:64, kernelSize:[3,3], activation:'relu', strides:[2,2]}).apply(input)
    // c = Convolution2D(32, 3, 3, activation='relu', )(c)
    c = tf.layers.conv2d({filters:32, kernelSize:[3,3], activation:'relu'}).apply(c)
    // c = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(c)
    c = tf.layers.conv2d({filters:64, kernelSize:[3,3], activation:'relu', padding:'same'}).apply(c)
    // c1 = MaxPooling2D((3, 3), strides=(2, 2))(c)
    let c1 = tf.layers.maxPooling2d({poolSize:[3,3], strides:[2,2]}).apply(c)
    // c2 = Convolution2D(96, 3, 3, activation='relu', subsample=(2, 2))(c)
    let c2 = tf.layers.conv2d({filters:96, kernelSize:[3,3], activation:'relu', strides:[2,2]}).apply(c)

    // m = merge([c1, c2], mode='concat', concat_axis=channel_axis)
    const m =  tf.layers.concatenate({axis:channel_axis}).apply([c1,c2])

    // c1 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(m)
    // c1 = Convolution2D(96, 3, 3, activation='relu', )(c1)
    c1 = tf.layers.conv2d({filters:64, kernelSize:[1,1], activation:'relu', padding:'same'}).apply(m)
    c1 = tf.layers.conv2d({filters:96, kernelSize:[3,3], activation:'relu'}).apply(c1)

    // c2 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(m)
    // c2 = Convolution2D(64, 7, 1, activation='relu', border_mode='same')(c2)
    // c2 = Convolution2D(64, 1, 7, activation='relu', border_mode='same')(c2)
    // c2 = Convolution2D(96, 3, 3, activation='relu', border_mode='valid')(c2)
    c2 = tf.layers.conv2d({filters:64, kernelSize:[1,1], activation:'relu', padding:'same'}).apply(m)
    c2 = tf.layers.conv2d({filters:64, kernelSize:[7,1], activation:'relu', padding:'same'}).apply(c2)
    c2 = tf.layers.conv2d({filters:64, kernelSize:[1,7], activation:'relu', padding:'same'}).apply(c2)
    c2 = tf.layers.conv2d({filters:96, kernelSize:[3,3], activation:'relu', padding:'valid'}).apply(c2)
    // m2 = merge([c1, c2], mode='concat', concat_axis=channel_axis)
    const m2 = tf.layers.concatenate({axis:channel_axis}).apply([c1,c2])

    // p1 = MaxPooling2D((3, 3), strides=(2, 2), )(m2)
    const p1 = tf.layers.maxPooling2d({poolSize:[3,3], strides:[2,2]}).apply(m2)
    // p2 = Convolution2D(192, 3, 3, activation='relu', subsample=(2, 2))(m2)
    const p2 = tf.layers.conv2d({filters:192, kernelSize:[3,3], activation:'relu', strides:[2,2]}).apply(m2)

    // m3 = merge([p1, p2], mode='concat', concat_axis=channel_axis)
    let m3 = tf.layers.concatenate({axis:channel_axis}).apply([p1,p2])
    // m3 = BatchNormalization(axis=channel_axis)(m3)
    m3 = tf.layers.batchNormalization({axis:channel_axis}).apply(m3)
    // m3 = Activation('relu')(m3)
    m3 = tf.layers.activation({activation: 'relu'}).apply(m3)
    return m3
}

// if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)
class lambLayer extends tf.layers.Layer {
    constructor() {
      super({});
    }
    // In this case, the output is a scalar.
    computeOutputShape(inputShape) { return []; }
   
    // call() is where we do the computation.
    call(input, kwargs) { return input*0.1;}
   
    // Every layer needs a unique name.
    getClassName() { return 'SquaredSum'; }
}

function inception_resnet_v2_A(input, scale_residual=True){
    // if K.image_dim_ordering() == "th":
    //     channel_axis = 1
    // else:
    //     channel_axis = -1
    let channel_axis = true ? -1 : 1

    //# Input is relu activation
    const init = input

    // ir1 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(input)
    let ir1 = tf.layers.conv2d({filters:32, kernelSize:[1,1], activation:'relu', padding:'same'}).apply(input)

    // ir2 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(input)
    // ir2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(ir2)
    let ir2 = tf.layers.conv2d({filters:32, kernelSize:[1,1], activation:'relu', padding:'same'}).apply(input)
    ir2 = tf.layers.conv2d({filters:32, kernelSize:[3,3], activation:'relu', padding:'same'}).apply(ir2)

    // ir3 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(input)
    // ir3 = Convolution2D(48, 3, 3, activation='relu', border_mode='same')(ir3)
    // ir3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(ir3)
    let ir3 = tf.layers.conv2d({filters:32, kernelSize:[1,1], activation:'relu', padding:'same'}).apply(input)
    ir3 = tf.layers.conv2d({filters:48, kernelSize:[3,3], activation:'relu', padding:'same'}).apply(ir3)
    ir3 = tf.layers.conv2d({filters:64, kernelSize:[3,3], activation:'relu', padding:'same'}).apply(ir3)

    // ir_merge = merge([ir1, ir2, ir3], concat_axis=channel_axis, mode='concat')
    const irMerge = tf.layers.concatenate({axis:channel_axis}).apply([ir1, ir2, ir3])

    // ir_conv = Convolution2D(384, 1, 1, activation='linear', border_mode='same')(ir_merge)
    let irConv = tf.layers.conv2d({filters:384, kernelSize:[1,1], activation:'linear', padding:'same'}).apply(irMerge)
    // if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)
    // console.log("bf:",irConv)
    irConv = scale_residual ? new lambLayer().apply(irConv) : irConv
    // console.log("after:",irConv)
    // out = merge([init, ir_conv], mode='sum')
    // out = BatchNormalization(axis=channel_axis)(out)
    // out = Activation("relu")(out)
    let out = tf.layers.add().apply([irConv, init])
    out = tf.layers.batchNormalization({axis:channel_axis}).apply(out)
    out = tf.layers.activation({activation: 'relu'}).apply(out)
    return out
}

function inception_resnet_v2_B(input, scale_residual=True){
    // if K.image_dim_ordering() == "th":
    //     channel_axis = 1
    // else:
    //     channel_axis = -1
    let channel_axis = true ? -1 : 1

    let init = input

    // ir1 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    let ir1 = tf.layers.conv2d({filters:192, kernelSize:[1,1], activation:'relu', padding:'same'}).apply(input)

    // ir2 = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(input)
    // ir2 = Convolution2D(160, 1, 7, activation='relu', border_mode='same')(ir2)
    // ir2 = Convolution2D(192, 7, 1, activation='relu', border_mode='same')(ir2)
    let ir2 = tf.layers.conv2d({filters:128, kernelSize:[1,1], activation:'relu', padding:'same'}).apply(input)
    ir2 = tf.layers.conv2d({filters:160, kernelSize:[1,7], activation:'relu', padding:'same'}).apply(ir2)
    ir2 = tf.layers.conv2d({filters:192, kernelSize:[7,1], activation:'relu', padding:'same'}).apply(ir2)

    // ir_merge = merge([ir1, ir2], mode='concat', concat_axis=channel_axis)
    const irMerge = tf.layers.concatenate({axis:channel_axis}).apply([ir1, ir2])

    // ir_conv = Convolution2D(1152, 1, 1, activation='linear', border_mode='same')(ir_merge)
    // if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)
    let irConv = tf.layers.conv2d({filters:1152, kernelSize:[1,1], activation:'linear', padding:'same'}).apply(irMerge)
    irConv = scale_residual ? new lambLayer().apply(irConv) : irConv

    // out = merge([init, ir_conv], mode='sum'),
    // out = BatchNormalization(axis=channel_axis)(out)
    // out = Activation("relu")(out)
    let out = tf.layers.add().apply([irConv, init])
    out = tf.layers.batchNormalization({axis:channel_axis}).apply(out)
    out = tf.layers.activation({activation: 'relu'}).apply(out)
    return out
}

function inception_resnet_v2_C(input, scale_residual=True){
    // if K.image_dim_ordering() == "th":
    //     channel_axis = 1
    // else:
    //     channel_axis = -1
    let channel_axis = true ? -1 : 1

    // # Input is relu activation
    let init = input

    // ir1 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    let ir1 = tf.layers.conv2d({filters:192, kernelSize:[1,1], activation:'relu', padding:'same'}).apply(input)

    // ir2 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    // ir2 = Convolution2D(224, 1, 3, activation='relu', border_mode='same')(ir2)
    // ir2 = Convolution2D(256, 3, 1, activation='relu', border_mode='same')(ir2)
    let ir2 = tf.layers.conv2d({filters:192, kernelSize:[1,1], activation:'relu', padding:'same'}).apply(input)
    ir2 = tf.layers.conv2d({filters:224, kernelSize:[1,3], activation:'relu', padding:'same'}).apply(ir2)
    ir2 = tf.layers.conv2d({filters:256, kernelSize:[3,1], activation:'relu', padding:'same'}).apply(ir2)

    // ir_merge = merge([ir1, ir2], mode='concat', concat_axis=channel_axis)
    const irMerge = tf.layers.concatenate({axis:channel_axis}).apply([ir1, ir2])

    // ir_conv = Convolution2D(2144, 1, 1, activation='linear', border_mode='same')(ir_merge)
    // if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)
    let irConv = tf.layers.conv2d({filters:2144, kernelSize:[1,1], activation:'linear', padding:'same'}).apply(irMerge)
    irConv = scale_residual ? new lambLayer().apply(irConv) : irConv

    // out = merge([init, ir_conv], mode='sum')
    // out = BatchNormalization(axis=channel_axis)(out)
    // out = Activation("relu")(out)
    let out = tf.layers.add().apply([irConv, init])
    out = tf.layers.batchNormalization({axis:channel_axis}).apply(out)
    out = tf.layers.activation({activation: 'relu'}).apply(out)
    return out
}

function reduction_A(input, k, l, m, n){
    // if K.image_dim_ordering() == "th":
    //     channel_axis = 1
    // else:
    //     channel_axis = -1
    let channel_axis = true ? -1 : 1

    // r1 = MaxPooling2D((3,3), strides=(2,2))(input)
    const r1 = tf.layers.maxPooling2d({poolSize:[3,3], strides:[2,2]}).apply(input)

    // r2 = Convolution2D(n, 3, 3, activation='relu', subsample=(2,2))(input)
    const r2 = tf.layers.conv2d({filters:n, kernelSize:[3,3], activation:'relu', strides:[2,2]}).apply(input)

    // r3 = Convolution2D(k, 1, 1, activation='relu', border_mode='same')(input)
    // r3 = Convolution2D(l, 3, 3, activation='relu', border_mode='same')(r3)
    // r3 = Convolution2D(m, 3, 3, activation='relu', subsample=(2,2))(r3)
    let r3 = tf.layers.conv2d({filters:k, kernelSize:[1,1], activation:'relu', padding:'same'}).apply(input)
    r3 = tf.layers.conv2d({filters:l, kernelSize:[3,3], activation:'relu', padding:'same'}).apply(r3)
    r3 = tf.layers.conv2d({filters:m, kernelSize:[3,3], activation:'relu', strides:[2,2]}).apply(r3)
    // m = merge([r1, r2, r3], mode='concat', concat_axis=channel_axis)
    let res = tf.layers.concatenate({axis:channel_axis}).apply([r1,r2,r3])
    // m = BatchNormalization(axis=1)(m)
    // m = Activation('relu')(m)
    res = tf.layers.batchNormalization({axis:1}).apply(res)
    res = tf.layers.activation({activation: 'relu'}).apply(res)
    return res
}

function reduction_resnet_v2_B(input){
    // if K.image_dim_ordering() == "th":
    //     channel_axis = 1
    // else:
    //     channel_axis = -1
    let channel_axis = true ? -1 : 1

    // r1 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(input)
    const r1 = tf.layers.maxPooling2d({poolSize:[3,3], strides:[2,2]}).apply(input)

    // r2 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    // r2 = Convolution2D(384, 3, 3, activation='relu', subsample=(2,2))(r2)
    let r2 = tf.layers.conv2d({filters:256, kernelSize:[1,1], activation:'relu', padding:'same'}).apply(input)
    r2 = tf.layers.conv2d({filters:384, kernelSize:[3,3], activation:'relu', strides:[2,2]}).apply(r2)

    // r3 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    // r3 = Convolution2D(288, 3, 3, activation='relu', subsample=(2, 2))(r3)
    let r3 = tf.layers.conv2d({filters:256, kernelSize:[1,1], activation:'relu', padding:'same'}).apply(input)
    r3 = tf.layers.conv2d({filters:288, kernelSize:[3,3], activation:'relu', strides:[2,2]}).apply(r3)

    // r4 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    // r4 = Convolution2D(288, 3, 3, activation='relu', border_mode='same')(r4)
    // r4 = Convolution2D(320, 3, 3, activation='relu', subsample=(2, 2))(r4)
    let r4 = tf.layers.conv2d({filters:256, kernelSize:[1,1], activation:'relu', padding:'same'}).apply(input)
    r4 = tf.layers.conv2d({filters:288, kernelSize:[3,3], activation:'relu', padding:'same'}).apply(r4)
    r4 = tf.layers.conv2d({filters:320, kernelSize:[3,3], activation:'relu', strides:[2,2]}).apply(r4)

    // m = merge([r1, r2, r3, r4], concat_axis=channel_axis, mode='concat')
    let m = tf.layers.concatenate({axis:channel_axis}).apply([r1, r2, r3, r4])
    // m = BatchNormalization(axis=channel_axis)(m)
    // m = Activation('relu')(m)
    m = tf.layers.batchNormalization({axis:channel_axis}).apply(m)
    m = tf.layers.activation({activation: 'relu'}).apply(m)
    return m
}

function create_inception_resnet_v2(nb_classes=1001, scale=true){
    // Creates a inception resnet v2 network

    // :param nb_classes: number of classes.txt
    // :param scale: flag to add scaling of activations
    // :return: Keras Model with 1 input (299x299x3) input shape and 2 outputs (final_output, auxiliary_output)

    // if K.image_dim_ordering() == 'th':
    //     init = Input((3, 299, 299))
    // else:
    //     init = Input((299, 299, 3))
    let init = true ?  tf.input({shape:[299,299,3]}) : tf.input({shape:[3,299,299]})
    
    // # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    // x = inception_resnet_stem(init)
    let x = inception_resnet_stem(init)

    // # 10 x Inception Resnet A
    // for i in range(10):
    //     x = inception_resnet_v2_A(x, scale_residual=scale)
    for( i of [...''.padEnd(10)].map((v,i)=>i)){
        x = inception_resnet_v2_A(x, scale)
    }

    // # Reduction A
    // x = reduction_A(x, k=256, l=256, m=384, n=384)
    x = reduction_A(x, 256, 256, 384, 384)
    console.log("reduction_A")
    // # 20 x Inception Resnet B
    // for i in range(20):
    //     x = inception_resnet_v2_B(x, scale_residual=scale)
    for( i of [...''.padEnd(10)].map((v,i)=>i)){
        x = inception_resnet_v2_B(x, scale)
    }
    console.log("inception_resnet_v2_B")
    // # Auxiliary tower
    // aux_out = AveragePooling2D((5, 5), strides=(3, 3))(x)
    // aux_out = Convolution2D(128, 1, 1, border_mode='same', activation='relu')(aux_out)
    // aux_out = Convolution2D(768, 5, 5, activation='relu')(aux_out)
    // aux_out = Flatten()(aux_out)
    // aux_out = Dense(nb_classes, activation='softmax')(aux_out)
    let aux_out = tf.layers.averagePooling2d({poolSize:[5,5], strides:[3,3]}).apply(x)
    aux_out = tf.layers.conv2d({filters:128, kernelSize:[1,1], activation:'relu', padding:'same'}).apply(aux_out)
    aux_out = tf.layers.conv2d({filters:768, kernelSize:[5,5], activation:'relu'}).apply(aux_out)
    aux_out = tf.layers.flatten().apply(aux_out)
    aux_out = tf.layers.dense({units:nb_classes, activation:'softmax'}).apply(aux_out)

    // # Reduction Resnet B
    x = reduction_resnet_v2_B(x)
    console.log("reduction_resnet_v2_B")
    // # 10 x Inception Resnet C
    // for i in range(10):
    //     x = inception_resnet_v2_C(x, scale_residual=scale)
    for( i of [...''.padEnd(10)].map((v,i)=>i)){
        x = inception_resnet_v2_C(x, scale)
    }
    console.log("inception_resnet_v2_C")
    // # Average Pooling
    // x = AveragePooling2D((8,8))(x)
    x = tf.layers.avgPooling2d({poolSize:[8,8]}).apply(x)

    // # Dropout
    // x = Dropout(0.8)(x)
    // x = Flatten()(x)
    x = tf.layers.dropout({rate:0.8}).apply(x)
    x = tf.layers.flatten().apply(x)

    // # Output
    // out = Dense(output_dim=nb_classes, activation='softmax')(x)
    let out = tf.layers.dense({units:nb_classes, activation:'softmax'}).apply(x)

    // model = Model(init, output=[out, aux_out], name='Inception-Resnet-v2')
    let model = tf.model({inputs:init, outputs:[out, aux_out], name:'Inception-Resnet-v2'})
    return model
}

exports.create_inception_resnet_v2 = create_inception_resnet_v2
