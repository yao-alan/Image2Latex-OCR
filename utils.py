import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

class Animator:
    """Class for graphing measurements."""
    def __init__(self, measurement_names, figsize=(9, 4), xlabel="n_epochs", 
                 ylabel='units', refresh=20):
        self.fig = plt.figure(figsize=figsize)
        self.t, self.y = 0, {}
        for m in measurement_names:
            self.y[m] = []
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.refresh = refresh

    # data is a list of tuples(metric_name, metric_value)
    def append(self, data):
        for name, m in data:
            self.y[name].append(m)
        self.t += 1
        if self.t % self.refresh == 0:
            self.update()

    def update(self):
        clear_output(wait=True)
        display(self.fig)
        
        plt.clf()
        for k in self.y.keys():
            plt.plot(np.arange(self.t), self.y[k], label=k)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.yscale('log', base=10)
        plt.title('Measurements')
        plt.legend()
        plt.grid()
        plt.show()


class MetricBuffer:
    """Class to easily accumulate different metrics (such as loss)."""
    # metrics = list of metric names
    def __init__(self, metric_names):
        self.metrics = {m: 0. for m in metric_names}
    
    # metrics = list of tuples(metric_name, value)
    def update(self, metrics):
        for metric_name, val in metrics:
            self.metrics[metric_name] += val

    def clear(self):
        self.metrics = {m: 0. for m in self.metrics.keys()}

    def __getitem__(self, key):
        return self.metrics[key]
