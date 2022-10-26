import numpy as np
import tensorflow as tf
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf  # 使用1版本
    tf.disable_v2_behavior()
    import tf_slim as slim
else:
    from tensorflow.contrib import slim

def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    with tf.variable_scope("deconv"):
        pool_size = 2

        deconv_filter = tf.Variable(
            tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

        deconv_output = tf.concat([deconv, x2], 3)
        deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


def gcblock(in_features,  out_channel, name):
    with tf.variable_scope(name):
        C = tf.shape(in_features)[3]
        batch_size = tf.shape(in_features)[0]
        in_features_trans = tf.transpose(tf.reshape(in_features, [batch_size, -1, C]), [0, 2, 1])  # B C HW

        channel_reduce = slim.conv2d(in_features, 1, [1, 1], activation_fn=None, scope='cr_conv')  # BHW1
        gs_att = tf.nn.softmax(tf.reshape(channel_reduce, [batch_size, -1, 1]), axis=1)  # B HW 1
        gs_agg = tf.matmul(in_features_trans, gs_att)  # B C 1
        gs_agg = tf.reshape(tf.transpose(gs_agg, [0, 2, 1]), [batch_size, 1, 1, out_channel])  # B 1 1 C
        gs_agg = slim.conv2d(gs_agg, out_channel // 4, [1, 1], activation_fn=lrelu, scope='gs_agg_conv1')
        #gs_agg = tf.contrib.layers.layer_norm(gs_agg, scope='layer_normalization')
        gs_agg = slim.conv2d(gs_agg, out_channel, [1, 1], activation_fn=None, scope='gs_agg_conv2')  # B 1 1 C

    return in_features + gs_agg


def pack_fits(im_fits):

    im = tf.maximum(im_fits, 0)
    im = tf.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    # out = tf.keras.backend.concatenate((im[0:H:2, 1:W:2, :],  # GBRG
    #                       im[0:H:2, 0:W:2, :],
    #                       im[1:H:2, 1:W:2, :],
    #                       im[1:H:2, 0:W:2, :]), axis=2)
    #  raw的一通道变四通道  H*W*1---->H/2* W/2* 4   红、绿、蓝、绿
    out = tf.keras.backend.concatenate((im[0:H:2, 0:W:2, :],
                                        im[0:H:2, 1:W:2, :],
                                        im[1:H:2, 1:W:2, :],
                                        im[1:H:2, 0:W:2, :]), axis=2)
    # out = tf.keras.backend.concatenate((im[1:H:2, 0:W:2, :],  # GBRG
    #                       im[1:H:2, 1:W:2, :],
    #                       im[0:H:2, 1:W:2, :],
    #                       im[0:H:2, 0:W:2, :]), axis=2)
    return out


def network(inputs):
    input_full = tf.expand_dims(pack_fits(inputs), axis=0)
    input_full = tf.minimum(input_full, 1.0)
    conv1 = slim.conv2d(input_full, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    conv1 = gcblock(conv1, 32, 'global_agg_1')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    #conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    conv2 = gcblock(conv2, 64, 'global_agg_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    #conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    conv3 = gcblock(conv3, 128, 'global_agg_3')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    conv4 = gcblock(conv4, 256, 'global_agg_4')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')
    conv5 = gcblock(conv5, 512, 'global_agg_5')
    #gap = tf.reduce_mean(conv5, axis=[1, 2], keepdims=True)  # Bx1x1xN


    #info_agg_conv4 = bgca(conv4, gap, 256, name='gap_agg_4')
    #up6 = upsample_and_concat(conv5, info_agg_conv4, 256, 512)
    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')
    #conv6 = global_agg(conv6, 256, 'global_agg_6')

    # info_agg_conv3 = bgca(conv3, gap, 128, name='info_agg_3')
    # up7 = upsample_and_concat(conv6, info_agg_conv3, 128, 256)
    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    #conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')
    #conv7 = global_agg(conv7, 128, 'global_agg_7')

    # info_agg_conv2 = bgca(conv2, gap, 64, name='info_agg_2')
    # up8 = upsample_and_concat(conv7, info_agg_conv2, 64, 128)
    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    #conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')
    #conv8 = global_agg(conv8, 64, 'global_agg_8')

    # info_agg_conv1 = bgca(conv1, gap, 32, name='info_agg_1')
    # up9 = upsample_and_concat(conv8, info_agg_conv1, 32, 64)
    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')
    #conv9 = global_agg(conv9, 32, 'global_agg_9')

    #conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv10_1')
    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=tf.nn.tanh, scope='g_conv10') * 0.58 + 0.52
    out = tf.depth_to_space(conv10, 2)
    output = out[0]
    output = tf.minimum(tf.maximum(output, 0), 1)

    output = output * 255

    output = tf.cast(output, tf.uint8)
    return output


if __name__ == '__main__':
    a = np.float32(np.random.random((1, 255, 255, 4)))
    b = network(a)

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        print('params: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        b = s.run(b)
        print(b.shape)