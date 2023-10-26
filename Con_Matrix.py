import matplotlib.pyplot as plt
import numpy as np

confusion_matrix = np.array([[0.81, 0.07, 0.09, 0.02, 0.00],
                            [0.05, 0.94, 0.01, 0.00, 0.00],
                            [0.07, 0.01, 0.92, 0.00, 0.00],
                            [0.09, 0.01, 0.01, 0.88, 0.00],
                            [0.21, 0.01, 0.24, 0.01, 0.53]])

categories = ['soil', 'bedrock', 'sand', 'big rock', 'background']

font = {'family': 'Times New Roman','size': 25}
plt.rc('font', **font)

fig, ax = plt.subplots(figsize=(10, 10))


im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)

cbar = ax.figure.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
cbar.ax.set_ylabel('Confusion Matrix Values', rotation=-90, va="bottom", fontsize=30)
cbar.ax.tick_params(labelsize=25)

ax.set_xticks(np.arange(len(categories)))
ax.set_yticks(np.arange(len(categories)))
ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=30)
ax.set_yticklabels(categories, fontsize=30)
ax.set_xlabel('Predicted Label',fontsize=30)
ax.set_ylabel('True Label',fontsize=30)

for i in range(len(categories)):
    for j in range(len(categories)):
        text = ax.text(j, i, confusion_matrix[i, j],
                       fontsize=30,ha="center", va="center", color="black")

plt.title("Confusion Matrix",fontsize='30')
plt.tight_layout()
plt.show()

