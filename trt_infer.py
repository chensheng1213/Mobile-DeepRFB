import sys
sys.path.append('../')

import common
import cv2
import configs
import time

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger()


def softmax(out_np, dim):
    s_value = np.exp(out_np) / np.sum(np.exp(out_np), axis=dim, keepdims=True)
    return s_value


class FaceClassify(object):
    def __init__(self, configs):
        self.engine_path = configs.face_classify_engine
        self.input_size = configs.classify_input_size
        self.image_size = self.input_size[1:]
        self.MEAN = configs.classify_mean
        self.STD = configs.classify_std
        self.engine = self.get_engine()
        self.context = self.engine.create_execution_context()


    def get_engine(self):

        f = open(self.engine_path, 'rb')
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())


    def detect(self, image_src, cuda_ctx = pycuda.autoinit.context):
        cuda_ctx.push()

        IN_IMAGE_H, IN_IMAGE_W = self.image_size


        img_in = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
        img_in = cv2.resize(img_in, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)

        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in /= 255.0


        mean0 = np.expand_dims(self.MEAN[0] * np.ones((IN_IMAGE_H, IN_IMAGE_W)), axis=0)
        mean1 = np.expand_dims(self.MEAN[1] * np.ones((IN_IMAGE_H, IN_IMAGE_W)), axis=0)
        mean2 = np.expand_dims(self.MEAN[2] * np.ones((IN_IMAGE_H, IN_IMAGE_W)), axis=0)
        mean = np.concatenate((mean0, mean1, mean2), axis=0)


        std0 = np.expand_dims(self.STD[0] * np.ones((IN_IMAGE_H, IN_IMAGE_W)), axis=0)
        std1 = np.expand_dims(self.STD[1] * np.ones((IN_IMAGE_H, IN_IMAGE_W)), axis=0)
        std2 = np.expand_dims(self.STD[2] * np.ones((IN_IMAGE_H, IN_IMAGE_W)), axis=0)
        std = np.concatenate((std0, std1, std2), axis=0)

        img_in = ((img_in - mean) / std).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)

        img_in = np.ascontiguousarray(img_in)

        self.context.active_optimization_profile = 0
        origin_inputshape = self.context.get_binding_shape(0)
        origin_inputshape[0], origin_inputshape[1], origin_inputshape[2], origin_inputshape[3] = img_in.shape
        self.context.set_binding_shape(0, (origin_inputshape))
        inputs, outputs, bindings, stream = common.allocate_buffers(self.engine, self.context)

        inputs[0].host = img_in
        trt_outputs = common.do_inference(self.context, bindings=bindings, inputs=inputs, outputs=outputs,
                                          stream=stream, batch_size=1)
        if cuda_ctx:
            cuda_ctx.pop()

        labels_sm = softmax(trt_outputs, dim=1)
        labels_max = np.argmax(labels_sm, axis=1)

        return labels_max.item()

