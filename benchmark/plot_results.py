def plot_comparisons(results={}, title="", x_label="", y_label="", output_path=""):
    import matplotlib.pyplot as plt
    import numpy as np

    # Flatten results: keys are (scenario, implementation), values are execution times or memory usage
    flat_results = {}
    for scenario, impl_times in results.items():
        for impl, time in impl_times.items():
            flat_results[(scenario, impl)] = time

    # Sort scenarios
    scenarios = sorted(set(k[0] for k in flat_results.keys()), key=lambda x: float(x) if isinstance(x, (int, float, str)) and str(x).replace('.','',1).isdigit() else str(x))
    implementations = sorted(set(k[1] for k in flat_results.keys()))

    # Prepare data for grouped bar plot
    bar_width = 0.2
    x = np.arange(len(scenarios))
    _, ax = plt.subplots(figsize=(10, 6))

    for idx, impl in enumerate(implementations):
        times = [flat_results.get((scenario, impl), 0) for scenario in scenarios]
        ax.bar(x + idx * bar_width, times, width=bar_width, label=impl)
        # ax.text(idx, value + 0.5, str(value), ha='center')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x + bar_width * (len(implementations)-1)/2)
    ax.set_xticklabels([str(s) for s in scenarios])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # Categories
    scales = ['0.5x', '1x', '1.5x', '2x']
    methods = ['YALMIP', 'SolveDB+', 'MPT']
    x_labels = [f"{scale}\n{method}" for scale in scales for method in methods]

    # Data: each list represents [Data I/O, Optimization, Model generation]
    data = {
        'YALMIP': [[0.1, 0.1, 2.5], [0.2, 0.2, 10.6], [0.3, 0.4, 49], [0.4, 0.6, 214]],
        'SolveDB+': [[0.05, 0.0, 0.5], [0.1, 0.0, 0.43], [0.2, 0.0, 0.6], [0.1, 0.0, 0.02]],
        'MPT': [[0.1, 0.1, 1.5], [0.2, 0.1, 3.0], [0.3, 0.1, 4.5], [0.4, 0.1, 6.6]]
    }

    # Prepare stacked values
    data_io = []
    optimization = []
    model_gen = []
    for scale_idx in range(len(scales)):
        for method in methods:
            io, opt, gen = data[method][scale_idx]
            data_io.append(io)
            optimization.append(opt)
            model_gen.append(gen)

    # X axis positions
    x = np.arange(len(x_labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot stacks
    p1 = ax.bar(x, data_io, width, label='Data I/O (of P4)', color='midnightblue')
    p2 = ax.bar(x, optimization, width, bottom=data_io, label='Optimization (of P4)', color='mediumseagreen')
    bottom_model = [i + j for i, j in zip(data_io, optimization)]
    p3 = ax.bar(x, model_gen, width, bottom=bottom_model, label='Model generation (of P4)', color='khaki')

    # Add total value on top
    totals = [round(io + opt + gen, 2) for io, opt, gen in zip(data_io, optimization, model_gen)]
    for i, total in enumerate(totals):
        ax.text(x[i], total + 0.5, str(total), ha='center', fontsize=8, fontweight='bold')

    # Customize plot
    ax.set_ylabel('Total execution time, sec')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_title('Execution Time by Method and Input Scale')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    plt.tight_layout()
    plt.show()
