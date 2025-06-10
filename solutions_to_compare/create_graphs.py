import re
import matplotlib.pyplot as plt
import numpy as np

def plot_characteristics():
    # Parse the results.txt file
    data = {}
    with open("solutions_to_compare/results.txt") as f:
        lines = f.readlines()

    # Grouping logic: base name (e.g. "GPOD") and variants (e.g. "GPOD (domain)")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Exclude unwanted entries
        if "all" in line or "including txt file" in line:
            current_label = None
            continue
        if not line.startswith("char") and not line.startswith("loc") and not line.startswith("eloc"):
            current_label = line
            data[current_label] = {}
        elif line.startswith("char") and current_label:
            m = re.search(r"char = (\d+)", line)
            if m:
                data[current_label]["char"] = int(m.group(1))
        elif line.startswith("eloc") and current_label:
            m = re.search(r"eloc = (\d+)", line)
            if m:
                data[current_label]["eloc"] = int(m.group(1))
        elif line.startswith("loc") and current_label:
            m = re.search(r"loc = (\d+)", line)
            if m:
                data[current_label]["loc"] = int(m.group(1))

    # Group by base name
    grouped = {}
    for label in data:
        if " (" in label:
            base, variant = label.split(" (", 1)
            variant = variant.rstrip(")")
        else:
            base = label
            variant = "base"
        if base not in grouped:
            grouped[base] = {}
        grouped[base][variant] = data[label]

    # Find all possible variants for consistent stacking order
    all_variants = set()
    for variants in grouped.values():
        all_variants.update(variants.keys())
    all_variants = sorted(all_variants, key=lambda v: (v != "base", v))

    # Use display names for bases
    base_display_names = {
        "GPOD": "GPO-D",
        # Add more mappings if needed
    }
    bases = list(grouped.keys())
    display_bases = [base_display_names.get(b, b) for b in bases]

    char_stacks = []
    eloc_stacks = []
    for variant in all_variants:
        char_stacks.append([grouped[base].get(variant, {}).get("char", 0) for base in bases])
        # Use eloc if available, otherwise fall back to loc (for GBOML)
        eloc_stacks.append([
            grouped[base].get(variant, {}).get("eloc",
                grouped[base].get(variant, {}).get("loc", 0)
            ) for base in bases
        ])

    # Colors for each variant
    colors = plt.get_cmap("tab20").colors
    variant_colors = {v: colors[i % len(colors)] for i, v in enumerate(all_variants)}

    variant_display_names = {
        "base": "Base",
        "domain": "With Domain",
        "txt file": "With txt setup file",
        # Add more mappings as needed
    }

    # Plot stacked bar for characters
    plt.figure(figsize=(10, 5))
    bottom = np.zeros(len(bases))
    for i, variant in enumerate(all_variants):
        plt.bar(display_bases, char_stacks[i], bottom=bottom, label=variant, color=variant_colors[variant])
        bottom += np.array(char_stacks[i])
    plt.ylabel("Characters")
    plt.title("Character Count for different microgrid implementations")
    plt.xticks(rotation=45, ha='right')
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = [variant_display_names.get(l, l) for l in labels]
    plt.legend(handles, labels, title="Measurement")
    plt.tight_layout()
    plt.savefig("char_counts_stacked.png")
    plt.show()

    # Plot stacked bar for effective lines of code (eloc or loc for GBOML)
    plt.figure(figsize=(10, 5))
    bottom = np.zeros(len(bases))
    for i, variant in enumerate(all_variants):
        plt.bar(display_bases, eloc_stacks[i], bottom=bottom, label=variant, color=variant_colors[variant])
        bottom += np.array(eloc_stacks[i])
    plt.ylabel("Effective Lines of Code")
    plt.title("Effective Lines of Code for different microgrid implementations")
    plt.xticks(rotation=45, ha='right')
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = [variant_display_names.get(l, l) for l in labels]
    plt.legend(handles, labels, title="Measurement")
    plt.tight_layout()
    plt.savefig("eloc_counts_stacked.png")
    plt.show()

if __name__ == "__main__":
    plot_characteristics()