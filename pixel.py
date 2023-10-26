import cv2
import numpy as np

image = cv2.imread("predict_out/mer/image/2n130900315eff1100p1901l0m1.jpg")
label = cv2.imread("predict_out/mer/label/2n130900315eff1100p1901l0m1.png")
orininal_h  = np.array(image).shape[0]
orininal_w  = np.array(image).shape[1]
classes_nums        = np.zeros([256], np.float32)
classes_nums += np.bincount(np.reshape(label, [-1]), minlength=256)
total_points_num    = orininal_h * orininal_w

print("打印像素点的值与数量。")
print('-' * 63)
print("| %25s | %15s |"%("Key", "Value"))
print('-' * 63)
for i in range(256):
    if classes_nums[i] > 0:
        print("| %25s | %15s |"%(str(i), str(classes_nums[i])))
        print('-' * 37)

