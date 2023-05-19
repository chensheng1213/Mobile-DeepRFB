#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary
import numpy as np

from nets.deeplabv3_plus import DeepLab

if __name__ == "__main__":
    input_shape     = [1024, 1024]
    num_classes     = 5
    backbone        = 'mobilenetv3_large'

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=16, pretrained=False).to(device)
    summary(model, (3, input_shape[0], input_shape[1]))

    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))

# import torch
# from torchvision.models.resnet import resnet101
#
# iterations = 300   # 重复计算的轮次
#
# model = resnet101()
# device = torch.device("cuda:0")
# model.to(device)
#
# random_input = torch.randn(1, 3, 224, 224).to(device)
# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#
# # GPU预热
# for _ in range(50):
#     _ = model(random_input)
#
# # 测速
# times = torch.zeros(iterations)     # 存储每轮iteration的时间
# with torch.no_grad():
#     for iter in range(iterations):
#         starter.record()
#         _ = model(random_input)
#         ender.record()
#         # 同步GPU时间
#         torch.cuda.synchronize()
#         curr_time = starter.elapsed_time(ender) # 计算时间
#         times[iter] = curr_time
#         # print(curr_time)
#
# mean_time = times.mean().item()
# print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))



