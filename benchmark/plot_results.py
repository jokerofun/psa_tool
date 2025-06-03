def plot_comparisons(results={}, title="", x_label="", y_label="", output_path=""):
    import matplotlib.pyplot as plt
    import numpy as np

    # Flatten results: keys are (scenario, implementation), values are execution times
    flat_results = {}
    for scenario, impl_times in results.items():
        for impl, time in impl_times.items():
            flat_results[(scenario, impl)] = time

    # Sort scenarios numerically if possible
    scenarios = sorted(set(k[0] for k in flat_results.keys()), key=lambda x: float(x) if isinstance(x, (int, float, str)) and str(x).replace('.','',1).isdigit() else str(x))
    implementations = sorted(set(k[1] for k in flat_results.keys()))

    # Prepare data for grouped bar plot
    bar_width = 0.2
    x = np.arange(len(scenarios))
    _, ax = plt.subplots(figsize=(10, 6))

    for idx, impl in enumerate(implementations):
        times = [flat_results.get((scenario, impl), 0) for scenario in scenarios]
        ax.bar(x + idx * bar_width, times, width=bar_width, label=impl)

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