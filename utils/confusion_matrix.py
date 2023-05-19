
'''
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵的函数
 
# 首先是从结果文件中读取预测标签与真实标签，然后将读取的标签信息传入python内置的混淆矩阵矩阵函数confusion_matrix(真实标签,
# 预测标签)中计算得到混淆矩阵，之后调用自己实现的混淆矩阵可视化函数plot_confusion_matrix()即可实现可视化。
# 三个参数分别是混淆矩阵归一化值，总的类别标签集合，可是化图的标题
 
def plot_confusion_matrix(cm, labels_name, title):
    np.set_printoptions(precision=2)
    # print(cm)
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
    # show confusion matrix
    plt.savefig('./fig/'+title+'.png', format='png')
 
gt = []
pre = []
with open("result.txt", "r") as f:
    for line in f:
        line=line.rstrip()#rstrip() 删除 string 字符串末尾的指定字符（默认为空格）
        words=line.split()
        pre.append(int(words[0]))
        gt.append(int(words[1]))
 
cm=confusion_matrix(gt,pre)  #计算混淆矩阵
print('type=',type(cm))
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]  #类别集合
plot_confusion_matrix(cm,labels,'confusion_matrix')  #绘制混淆矩阵图，可视化
'''
import numpy as np
import itertools
import matplotlib.pyplot as plt

'''
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        matrix = cm
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.figure()
    # 设置输出的图片大小
    figsize = 10, 8
    figure, ax = plt.subplots(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # 设置title的大小以及title的字体
    font_title = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 15,
                  }
    plt.title(title, fontdict=font_title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, )
    plt.yticks(tick_marks, classes)
    # 设置坐标刻度值的大小以及刻度值的字体
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    print(labels)
    [label.set_fontname('Times New Roman') for label in labels]
    if normalize:
        fm_int = 'd'
        fm_float = '.3%'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fm_float),
                     horizontalalignment="center", verticalalignment='bottom', family="Times New Roman",
                     weight="normal", size=15,
                     color="white" if cm[i, j] > thresh else "black")
            plt.text(j, i, format(matrix[i, j], fm_int),
                     horizontalalignment="center", verticalalignment='top', family="Times New Roman", weight="normal",
                     size=15,
                     color="white" if cm[i, j] > thresh else "black")
    else:
        fm_int = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fm_int),
                     horizontalalignment="center", verticalalignment='bottom',
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    # 设置横纵坐标的名称以及对应字体格式
    font_lable = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 15,
                  }

    plt.ylabel('True label', font_lable)
    plt.xlabel('Predicted label', font_lable)

    # plt.savefig('confusion_matrix.eps', dpi=800, format='eps')
    plt.savefig('confusion_matrix.png', dpi=800, format='png')
    plt.show()

#69.09
#69.93
cnf_matrix = np.array([[1233175296, 112464430, 139915129, 37041363, 5970826],
                       [35058113, 670198022, 5586642, 1822992, 62744],
                       [66133383, 8679845, 886461222, 2237173, 1308456],
                       [20771582, 3450900, 2514847, 203765506, 25643],
                       [3490309, 132217, 3928602, 118138, 8647388]])
68.25
cnf_matrix = np.array([[1249991866, 103705804, 130324740, 36685093, 7859541],
                       [47592691, 656416400, 5083312, 3591299, 44811],
                       [86354727, 8798822, 865781355, 1870753, 2014422],
                       [24578869, 2474660, 2551285, 200860893, 62771],
                       [3053888, 192391, 3504516, 68218, 9497641]])
6993
cnf_matrix = np.array([[1258879724, 92065262, 141641375, 29454871, 6525812],
                       [49478536, 655540953, 5792771, 1866425, 49828],
                       [67269156, 4974337, 889756411, 1386505, 1433670],
                       [27764483, 2419942, 3591817, 196667907, 84329],
                       [3377101, 85611, 3133775, 46197, 9673970]])
cnf_matrix = np.array([[1663747066, 121426501, 190519007, 42224962, 9971735],
                       [67006330, 855650652, 6659084, 1545812, 35473],
                       [85771813, 4172995, 1165968629, 1910044, 1456372],
                       [30804336, 2798224, 3487849, 268824390, 86968],
                       [4811084, 193985, 3740848, 89694, 15818835]])
# attack_types = ['泥土', '基岩', '沙子', '大岩石', '背景']
# plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=True, title='混淆矩阵')
attack_types = ['soil', 'bedrock', 'sand', 'big rock', 'background']
plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=True, title='Confusion matrix')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
'''
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     confusion = np.array([[1265368603, 91518638, 135377852, 30837621, 5464330],[48483339, 658452371, 4160065, 1600239, 32499],
#                           [63917692, 4963692, 893290738, 1458541, 1189416],[25876837, 1897978, 2584173, 200131067, 38423],
#                           [4078491, 123360, 3120840, 87554, 8906409]])#//混淆矩阵
#     # 热度图，后面是指定的颜色块，可设置其他的不同颜色
#     plt.imshow(confusion, cmap=plt.cm.Reds)
#     # ticks 坐标轴的坐标点
#     # label 坐标轴标签说明
#     indices = range(len(confusion))
#     # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
#     # plt.xticks(indices, [0, 1, 2])
#     # plt.yticks(indices, [0, 1, 2])
#     plt.xticks(indices, ['泥土', '基岩', '沙子', '大岩石', '背景'])
#     plt.yticks(indices, ['泥土', '基岩', '沙子', '大岩石', '背景'])
#
#     plt.colorbar()
#
#     plt.xlabel('预测值')
#     plt.ylabel('真实值')
#     plt.title('混淆矩阵')
#
#     # plt.rcParams两行是用于解决标签不能显示汉字的问题
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.rcParams['axes.unicode_minus'] = False
#
#     # 显示数据
#     for first_index in range(len(confusion)):  # 第几行
#         for second_index in range(len(confusion[first_index])):  # 第几列
#             plt.text(first_index, second_index, confusion[first_index][second_index])
#     # 在matlab里面可以对矩阵直接imagesc(confusion)
#     # 显示
#     plt.show()
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
                                show_normed=True)
plt.show()













