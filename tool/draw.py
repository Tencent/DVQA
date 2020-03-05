import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def mos_scatter(mos, pred, dataset_name):
    plt.scatter(mos, pred, alpha=0.6)
    plt.xlabel('MOS')
    plt.ylabel('PRED')
    plt.title(dataset_name)
    plt.plot([0, 100], [0, 100])
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.savefig('{}.png'.format(dataset_name), dpi=1000)



