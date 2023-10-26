import numpy as np
import itertools
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
multiclass = np.array([[1663747066, 121426501, 190519007, 42224962, 9971735],
                       [67006330, 855650652, 6659084, 1545812, 35473],
                       [85771813, 4172995, 1165968629, 1910044, 1456372],
                       [30804336, 2798224, 3487849, 268824390, 86968],
                       [4811084, 193985, 3740848, 89694, 15818835]])
class_names = ['soil', 'bedrock', 'sand', 'big rock', 'background']
fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                colorbar=True,
                                show_absolute=False,
                                show_normed=True,
                                class_names=class_names)
plt.figure(figsize=(8, 8))
plt.show()













