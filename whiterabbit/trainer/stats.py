# -*- coding: utf-8 -*-
"""
White Rabbit Chess Engine.

Trainer statistics with matplotlib.
"""
from statistics import mean, median

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def func_update_stats(self) -> None:
    """
    Update stats.

    Uploads beautiful graphics.
    """
    # Update values
    serie: list[int] = []
    for _, result in self.scores.items():
        serie.append(result)

    mean_value: float = float(mean(serie))
    median_value: float = float(median(serie))
    severity_value: float = float(max(serie) - min(serie))

    self.stats_graph["med"].append(median_value)
    self.stats_graph["mea"].append(mean_value)
    self.stats_graph["ete"].append(severity_value)

    # Plot values
    plt.figure(figsize=(9, 3))

    plt.subplot(131)
    plt.plot(range(len(self.stats_graph["med"])), self.stats_graph["med"])
    """
    plt.xlabel("Iteration")
    plt.ylabel("Median")
    """
    plt.title("Median")

    plt.subplot(132)
    plt.plot(range(len(self.stats_graph["mea"])), self.stats_graph["mea"])
    """
    plt.xlabel("Iteration")
    plt.ylabel("Mean")
    """
    plt.title("Mean")

    plt.subplot(133)
    plt.plot(range(len(self.stats_graph["ete"])), self.stats_graph["ete"])
    """
    plt.xlabel("Iteration")
    plt.ylabel("Severity")
    """
    plt.title("Severity")

    """
    plt.suptitle("Training statistics")
    """

    plt.savefig("data/training/graph.png")
    plt.close()
