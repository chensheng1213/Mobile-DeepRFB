import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import torch

BATCH_SIZE = 32


USE_FP16 = True
target_dtype = np.float16 if USE_FP16 else np.float32


f = open("resnet_engine.trt", "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()


input_batch = np.random.randn(BATCH_SIZE, 224, 224, 3).astype(target_dtype)
output = np.empty([BATCH_SIZE, 1000], dtype = target_dtype)

d_input = cuda.mem_alloc(1 * input_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()

def predict(batch):

    cuda.memcpy_htod_async(d_input, batch, stream)
    context.execute_async_v2(bindings, stream.handle, None)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()

    return output


def preprocess_input(input):

    result = torch.from_numpy(input).transpose(0,2).transpose(1,2)
    return np.array(result, dtype=target_dtype)
preprocessed_inputs = np.array([preprocess_input(input) for input in input_batch])
print("Warming up...")
pred = predict(preprocessed_inputs)
print("Done warming up!")

t0 = time.time()
pred = predict(preprocessed_inputs)
t = time.time() - t0
print("Prediction cost {:.4f}s".format(t))