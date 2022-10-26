# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from PIL import Image
from PySide2.QtCore import QThread, Signal, QObject
from threading import Lock
from time import sleep

if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf  # 使用1版本

    tf.disable_v2_behavior()

from gc_unet import network

'''
获取并处理数据
'''
checkpoint_dir = "./checkpoint/slim_gc_unet_5gc/"


class ProcessDone(QObject):
    OnProcessDone = Signal(object)


lock = Lock()


class Consumer(QThread):
    def __init__(self, parent, queue):
        super(Consumer, self).__init__(parent)
        # 参数初始化
        self.v = 0  # 亮度滑动平均
        self.beta = 0.9  # 衰减度
        self.ratio_ = 1  # 增强比率
        self.t = 1  # 时间系数 用于平滑衰减
        self.queue = queue  # 输入队列
        self.thr = 0.3
        # 创建session
        config = tf.ConfigProto()  # allow_soft_placement=True
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.in_image = tf.placeholder(tf.float32, [None, None])
        self.out_image = network(self.in_image)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt:
            print('loaded ' + ckpt.model_checkpoint_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("初始化完毕！")
        self.processDone = ProcessDone()

    def set_thr(self, thr):
        # 修改阈值
        with lock:
            self.thr = thr

    def adjust(self, img):
        # 滑动平均自适应
        mm = np.mean(img)
        m = self.beta * self.v + (1 - self.beta) * mm
        m = m / (1 - self.beta ** self.t)
        self.v = m
        self.t += 1
        print('mean_val:', mm, "fuse val:", m, "thr:", self.thr)
        return int(self.thr / m)

    def run(self):
        while not self.isInterruptionRequested():
            # print(self.isInterruptionRequested())
            try:
                input_data = self.queue.get_nowait()
                self.ratio_ = np.maximum(self.adjust(input_data), 0.1)
                print('ratio:', self.ratio_)
                im = input_data * self.ratio_
                output = self.sess.run(self.out_image, feed_dict={self.in_image: im})
                output = Image.fromarray(output).toqpixmap()
                self.processDone.OnProcessDone.emit(output)
            except:
                sleep(0.02)
