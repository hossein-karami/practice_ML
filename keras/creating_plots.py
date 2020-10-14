import numpy as np
import itertools
import matplotlib.pyplot as plt



def plot_confusion_matrix(
    conf_matrix,
    classes,
    normalize=False,
    title="Confusion Matrix",
    cmap=plt.cm.Blues
    ):

    plt.imshow(conf_matrix, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        conf_matrix.as_type("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized Confution Matrix")
    else:
        print("Confusion Matrix, without normalization")
    
    print(conf_matrix)
    threshold = conf_matrix.max() / 2.0
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(
            j, i, conf_matrix[i, j],
            horizontalalignment = "center",
            color="white" if conf_matrix[i, j] > threshold else "black"
        )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Prediction label")
    #plt.show()