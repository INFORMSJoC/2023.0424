from bisect import bisect
from matplotlib import pyplot as plt
import numpy as np

# metric is a list of lists with dimensions num_instances * num_methods
# methods is a list of length num_methods
# time_values is a list of length num_tau
def solution_profile(metric, methods, time_values, file_name = "", title_name = ""):
    num_instances = len(metric)
    num_methods = len(methods)
    num_tau = len(time_values)
    
    sorted_metric = [[0 for i in range(num_instances)] for m in range(num_methods)]
    for m in range(num_methods):
        for i in range(num_instances):
            sorted_metric[m][i] = metric[i][m]

    for m in range(num_methods):
        sorted_metric[m].sort()
    
    sorted_metric_dist = [[0 for t in range(num_tau)] for m in range(num_methods)]
    for m in range(num_methods):
        for t in range(num_tau):
            if t == 0:
                sorted_metric_dist[m][t] = bisect(sorted_metric[m],time_values[t])
            else:
                sorted_metric_dist[m][t] = bisect(sorted_metric[m],time_values[t],lo=sorted_metric_dist[m][t-1])
    
    for m in range(num_methods):
        for t in range(num_tau):
            sorted_metric_dist[m][t] /= (num_instances/100.0)


    font_size = 28
    line_width = 4
    fig_size = (10,8)
    dpi_val = 1200

    plt.figure(figsize=fig_size, dpi=dpi_val)
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams.update({'font.size': font_size})

    for m in range(num_methods):
        if m == 1:
            plt.plot(time_values,sorted_metric_dist[m],label=methods[m],linewidth=4.0,linestyle='dashed')
        elif m == 2:
            plt.plot(time_values,sorted_metric_dist[m],label=methods[m],linewidth=4.0,linestyle=':')
        else:
            plt.plot(time_values,sorted_metric_dist[m],label=methods[m],linewidth=4.0)
    plt.legend()
    plt.xscale('log')
    plt.ylabel('% instances solved within time T')
    plt.xlabel('Time T (seconds)')
    plt.title(title_name)
    plt.grid(lw=0.1)
    plt.xticks([2, 5, 20, 50, 200, 500, 2000, 7200],['2','5','20','50','200  ','500','2000 ', '7200'])

    plt.subplots_adjust(left=0.14,right=0.98,bottom=0.12,top=0.94)
    
    if file_name:
        plt.savefig(file_name + ".eps", format='eps')