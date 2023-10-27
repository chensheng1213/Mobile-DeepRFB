import numpy as np
import matplotlib.pyplot as plt

data = {
    '(a)': np.array([
        [0.81, 0.07, 0.09, 0.02, 0.01],
        [0.05, 0.94, 0.01, 0, 0],
        [0.02, 0.03, 0.92, 0.02, 0.01],
        [0, 0, 0.01, 0.98, 0.01],
        [0.21, 0.01, 0.24, 0.01, 0.53]
    ]),
    '(b)': np.array([
        [0.82, 0.07, 0.09, 0.02, 0.0],
        [0.07, 0.92, 0.01, 0.0, 0.0],
        [0.09, 0.01, 0.90, 0.0, 0.0],
        [0.11, 0.01, 0.01, 0.87, 0.0],
        [0.19, 0.01, 0.21, 0.01, 0.58]
    ]),
    '(c)': np.array([
        [0.82, 0.06, 0.09, 0.02, 0.01],
        [0.07, 0.92, 0.01, 0.0, 0.0],
        [0.07, 0.01, 0.92, 0.0, 0.0],
        [0.12, 0.01, 0.02, 0.85, 0.0],
        [0.21, 0.01, 0.19, 0.0, 0.59]
    ]),
    '(d)': np.array([
        [0.82, 0.06, 0.09, 0.02, 0.01],
        [0.07, 0.92, 0.01, 0.0, 0.0],
        [0.07, 0.0, 0.93, 0.0, 0.0],
        [0.10, 0.01, 0.01, 0.88, 0.0],
        [0.20, 0.01, 0.15, 0.0, 0.64]
    ])
    # Add data for (b), (c), and (d) similarly
}

labels = ['soil', 'bedrock', 'sand', 'big rock', 'background']

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

font = {'family': 'Times New Roman','size': 19}
plt.rc('font', **font)
for ax, key in zip(axes.ravel(), data.keys()):
    conf_matrix = data[key]
    cax = ax.matshow(conf_matrix, cmap='Blues')
    #ax.set_title(key if key != '' else '', fontdict=font_properties)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, fontsize=17)
    ax.set_yticklabels(labels, fontsize=17)
    ax.tick_params(direction='in')

    ax.set_xlabel('Predicted Label', fontsize=17)
    ax.set_ylabel('True Label', fontsize=17)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{conf_matrix[i, j]:.2f}', ha='center', va='center', color='k')

    ax.annotate(key, xy=(0.5, -0.18), xycoords='axes fraction', ha='center', va='bottom')

plt.tight_layout(pad=0)
plt.savefig('confusion_matrix.png', format='png', dpi=600)
plt.show()
