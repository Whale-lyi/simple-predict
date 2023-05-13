from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    生成混淆矩阵并保存为图片

    参数：
    y_true：实际类别，类型为一维数组或列表
    y_pred：预测类别，类型为一维数组或列表
    class_names：类别名称，类型为列表
    save_path：图片保存路径，类型为字符串

    返回值：
    无返回值
    """

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 绘制混淆矩阵图
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion matrix')
    plt.colorbar()

    # 设置x轴和y轴刻度及标签
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # 在矩阵中填充数字标签
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # 保存混淆矩阵图
    plt.savefig(save_path)
