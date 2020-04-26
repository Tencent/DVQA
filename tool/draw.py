import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def mos_scatter(mos, pred):
    
    fig = plt.figure(figsize=(48,48))
    plt.scatter(mos, pred, alpha=0.8)
    plt.xlabel('MOS')
    plt.ylabel('PRED')
    plt.plot([0, 100], [0, 100])

    return fig


