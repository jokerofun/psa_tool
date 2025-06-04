def plot_segmented_bar_chart(results, title, x_label, y_label, output_path):
    """
    Plots a segmented (stacked) bar chart for grouped results.
    Each group is a key in results, each group has 4 bars (inner keys), and each bar is segmented by the list values.
    The legend is based on segment index: 0="setup", 1="dataflow tasks", 2="optimization".
    Adds space between bars in a group, more space between groups, black border around bars, and names above bars with sum.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Define segment colors and labels
    segment_colors = ["#f52626", "#f28e2b", "#4ea7e2"]
    segment_labels = ["setup", "dataflow tasks", "optimization"]

    groups = list(results.keys())
    bars = list(next(iter(results.values())).keys())  # assumes all groups have same inner keys
    n_groups = len(groups)
    n_bars = len(bars)
    n_segments = len(segment_labels)

    # Prepare data: shape (n_groups, n_bars, n_segments)
    data = np.zeros((n_groups, n_bars, n_segments))
    for i, group in enumerate(groups):
        for j, bar in enumerate(bars):
            values = results[group][bar]
            for k in range(min(len(values), n_segments)):
                data[i, j, k] = values[k]

    bar_width = 0.18
    bar_spacing = 0.10  # space between bars within a group
    group_spacing = 0.40  # more space between groups
    group_width = n_bars * bar_width + (n_bars - 1) * bar_spacing + group_spacing
    x = np.arange(n_groups) * group_width

    fig, ax = plt.subplots(figsize=(12, 7))

    for j, bar in enumerate(bars):
        bottom = np.zeros(n_groups)
        for k in range(n_segments):
            segment = data[:, j, k]
            ax.bar(x + j * (bar_width + bar_spacing), segment, bar_width, bottom=bottom,
                   color=segment_colors[k],
                   label=segment_labels[k] if (j == 0) else "",
                   edgecolor='black', linewidth=1)
            bottom += segment

        # Add name and sum above each bar (2 lines)
        for i in range(n_groups):
            bar_height = bottom[i]
            bar_sum = np.sum(data[i, j, :])
            label_text = f"{bars[j]}\n{bar_sum:.2f}"
            ax.text(x[i] + j * (bar_width + bar_spacing), bar_height + 0.5, label_text,
                    ha='center', va='bottom', fontsize=9, rotation=0, fontweight='bold')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x + ((n_bars-1)*(bar_width + bar_spacing))/2)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def plot_bar_chart(results, title, x_label, y_label, output_path):
    """
    Plots a grouped bar chart for results with one value per bar.
    Each group is a key in results, each group has 4 bars (inner keys), each bar has a different color.
    The legend describes the colors for "GPO-D", "CVXPY", "Pyomo", and "GBOML".
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Define bar colors and labels
    bar_colors = {
        "GPO-D": "#4e79a7",
        "CVXPY": "#f28e2b",
        "Pyomo": "#76b7b2",
        "GBOML": "#e15759"
    }
    bars = ["GPO-D", "CVXPY", "Pyomo", "GBOML"]
    groups = list(results.keys())
    n_groups = len(groups)
    n_bars = len(bars)

    # Prepare data: shape (n_groups, n_bars)
    data = np.zeros((n_groups, n_bars))
    for i, group in enumerate(groups):
        for j, bar in enumerate(bars):
            value = results[group][bar]
            data[i, j] = value

    bar_width = 0.18
    bar_spacing = 0.10  # space between bars within a group
    group_spacing = 0.40  # more space between groups
    group_width = n_bars * bar_width + (n_bars - 1) * bar_spacing + group_spacing
    x = np.arange(n_groups) * group_width

    fig, ax = plt.subplots(figsize=(12, 7))

    for j, bar in enumerate(bars):
        ax.bar(x + j * (bar_width + bar_spacing), data[:, j], bar_width,
               color=bar_colors[bar],
               label=bar,
               edgecolor='black', linewidth=1)
        # Add name and value above each bar (2 lines)
        for i in range(n_groups):
            bar_height = data[i, j]
            label_text = f"{bar}\n{bar_height:.2f}"
            ax.text(x[i] + j * (bar_width + bar_spacing), bar_height + 0.5, label_text,
                    ha='center', va='bottom', fontsize=9, rotation=0, fontweight='bold')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x + ((n_bars-1)*(bar_width + bar_spacing))/2)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    results_exec_time_T = {
        "24 x 1" : {
            "GPO-D" : [0.0, 5.71, 1.04],
            "CVXPY": [0.0, 5.72, 1.02],
            "Pyomo": [0.0, 5.69, 0.10],
            "GBOML": [0.0, 6.56, 0.25],
        },
        "24 x 2" : {
            "GPO-D" : [0.0, 5.74, 2.15],
            "CVXPY": [0.0, 5.67, 2.13],
            "Pyomo": [0.0, 5.74, 0.16],
            "GBOML": [0.0, 6.33, 0.39],
        },
        "24 x 3": {
            "GPO-D" : [0.0, 5.66, 3.17],
            "CVXPY": [0.0, 5.62, 3.14],
            "Pyomo": [0.0, 5.69, 0.23],
            "GBOML": [0.0, 6.47, 0.29],
        },
        "24 x 5": {
            "GPO-D" : [0.0, 5.63, 5.45],
            "CVXPY": [0.0, 5.60, 5.37],
            "Pyomo": [0.0, 5.66, 0.39],
            "GBOML": [0.0, 6.48, 0.30],
        }
    }
    reults_memory_usage_T = {
        "24 x 1" : {
            "GPO-D" : 292.24,
            "CVXPY": 294.57,
            "Pyomo": 297.05,
            "GBOML": 260.25,
        },
       "24 x 2" : {
            "GPO-D" : 316.07,
            "CVXPY": 324.1,
            "Pyomo": 325.29,
            "GBOML": 261.19,
        },
        "24 x 3" : {
            "GPO-D" : 347.29,
            "CVXPY": 354.52,
            "Pyomo": 348.48,
            "GBOML": 264.15,
        },
        "24 x 5" : {
            "GPO-D" : 393.45,
            "CVXPY": 401.39,
            "Pyomo": 366.43,
            "GBOML": 267.21,
        },
    }
    results_exec_time_Homes = {
        "50" : {
            "GPO-D" : [0.0, 5.61, 1.04],
            "CVXPY": [0.0, 5.65, 1.02],
            "Pyomo": [0.0, 5.60, 0.13],
            "GBOML": [0.0, 6.64, 0.24],
        },
        "100" : {
            "GPO-D" : [0.0, 11.31, 1.99],
            "CVXPY": [0.0, 11.23, 1.95],
            "Pyomo": [0.0, 11.33, 0.14],
            "GBOML": [0.0, 13.32, 0.38],
        },
        "200": {
            "GPO-D" : [0.01, 22.82, 3.60],
            "CVXPY": [0.0, 22.49, 3.52],
            "Pyomo": [0.0, 22.54, 0.22],
            "GBOML": [0.0, 26.83, 0.70],
        },
        "500": {
            "GPO-D" : [0.02, 58.76, 9.01],
            "CVXPY": [0.0, 56.78, 9.06],
            "Pyomo": [0.0, 58.68, 0.55],
            "GBOML": [0.0, 0.0, 0.0],
        },
        "1000": {
            "GPO-D" : [0.03, 122.35, 20.99],
            "CVXPY": [0.0, 121.79, 20.40],
            "Pyomo": [0.0, 120.05, 0.98],
            "GBOML": [0.0, 0.0, 0.0],
        }
    }
    results_memory_usage_Homes = {
         "50" : {
            "GPO-D" : 355.43,
            "CVXPY": 353.21,
            "Pyomo": 351.45,
            "GBOML": 266.67,
        },
       "100" : {
            "GPO-D" : 351.54,
            "CVXPY": 350.17,
            "Pyomo": 348.68,
            "GBOML": 269.05,
        },
        "200" : {
            "GPO-D" : 366.04,
            "CVXPY": 368.57,
            "Pyomo": 366.37,
            "GBOML": 278.44,
        },
        "500" : {
            "GPO-D" : 398.66,
            "CVXPY": 410.23,
            "Pyomo": 386.31,
            "GBOML": 0,
        },
         "1000" : {
            "GPO-D" : 462.78,
            "CVXPY": 469.37,
            "Pyomo": 412.99,
            "GBOML": 0,
        },
    }

    plot_segmented_bar_chart(results=results_exec_time_T,
                     title="Average execution time for different microgrid implementations", 
                     x_label="Time length for microgrid problem in hours (with 50 homes and solar panels)", 
                     y_label="Average execution time in seconds", 
                     output_path="figures/exec_time_comparison_scaleT.png")
    
    plot_segmented_bar_chart(results=results_exec_time_Homes,
                     title="Average execution time for different microgrid implementations", 
                     x_label="Number of homes and solar panels for microgrid problem (for 24 hours)", 
                     y_label="Average execution time in seconds", 
                     output_path="figures/exec_time_comparison_scaleHomes.png")
    
    plot_bar_chart(results=reults_memory_usage_T, 
                     title="Average memory usage for different microgrid implementations",
                     x_label="Time length for microgrid problem in hours (with 50 homes and solar panels)",
                     y_label="Average memory usage in MiB",
                     output_path="figures/memory_usage_comparison_scaleT.png")

    plot_bar_chart(results=results_memory_usage_Homes, 
                     title="Average memory usage for different microgrid implementations",
                     x_label="Number of homes and solar panels for microgrid problem (for 24 hours)",
                     y_label="Average memory usage in MiB",
                     output_path="figures/memory_usage_comparison_scaleHomes.png")
