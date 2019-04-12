import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

numbers_string = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def plot_confusion_matrix(confusion_matrix, title, save_name):
    df_cm = pd.DataFrame(confusion_matrix, index=numbers_string, columns=numbers_string)
    ax = plt.axes()
    cm_heatmap = sn.heatmap(df_cm, ax=ax, annot=True, cmap='Blues', fmt='g')
    ax.set_title(title)
    figure = cm_heatmap.get_figure()
    figure.savefig(save_name, dpi=400)
    plt.show()