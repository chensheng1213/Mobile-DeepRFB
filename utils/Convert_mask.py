import numpy as np
import PIL.Image as Image

src= Image.open("D:/Pycharm_AutoDL/deeplabv3-plus-pytorch/test_img/NLB_436560164EDR_F0220000NCAM00259M1.png")
mat = np.array(src)
mat = mat.astype(np.uint8)
dst = Image.fromarray(mat, 'RGB')
bin_colormap = [0,0,0] + [255,255,255]*254    # 二值调色板
dst.putpalette(bin_colormap)
dst.save('new.png')
src.close()
