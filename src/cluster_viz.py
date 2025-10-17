import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import os
import json
from tqdm import tqdm

import sys

aspect_all = 0.5

def subscript_numbers(formula):
    sub_numbers = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return formula.translate(sub_numbers)


def load_onset_edge_data():
    """
    Load onset edge data from onset_edge.json (generated from onset-based analysis)
    """
    onset_file = "onset_edge.json"

    if not os.path.exists(onset_file):
        print(f"FATAL ERROR: {onset_file} not found.")
        print("Please Copy data/onset_edge.json .")
        print("Program terminating immediately.")
        exit(1)

    try:
        with open(onset_file, 'r') as f:
            onset_data = json.load(f)
    except Exception as e:
        print(f"FATAL ERROR: Failed to load {onset_file}")
        print(f"Error: {e}")
        print("Program terminating immediately.")
        exit(1)

    print(f"✓ Loaded onset edge data for {len(onset_data)} materials")
    return onset_data


def load_clustering_data(extract_layer):
    """
    Load pre-computed clustering data from file
    """
    clustering_file = f"clustering/{extract_layer}/clustering_data.npz"

    if not os.path.exists(clustering_file):
        print(f"FATAL ERROR: Clustering data file not found: {clustering_file}")
        print("Please run the clustering script first to generate the linkage matrix.")
        print("Program terminating immediately.")
        exit(1)

    try:
        clustering_data = np.load(clustering_file)
        linkage_matrix = clustering_data["linkage_matrix"]
        print(f"✓ Loaded pre-computed linkage matrix from {clustering_file}")
        print(f"  Linkage matrix shape: {linkage_matrix.shape}")
        return linkage_matrix
    except Exception as e:
        print(f"FATAL ERROR: Failed to load clustering data from {clustering_file}")
        print(f"Error: {e}")
        print("Program terminating immediately.")
        exit(1)


def create_condensed_dendrogram(linkage_matrix, cluster_labels, structure_names, n_clusters, output_dir):
    """
    Create cluster-level dendrogram with each cluster as a leaf
    """

    # Calculate cluster centroids
    df = pd.read_json("features/features.json")
    spectrum_data = np.array(df['prediction'].tolist())

    cluster_centroids = []
    cluster_info = []

    for i in range(1, n_clusters + 1):
        cluster_mask = cluster_labels == i
        cluster_spectra = spectrum_data[cluster_mask]
        cluster_structures = structure_names[cluster_mask]

        if len(cluster_spectra) > 0:
            centroid = np.mean(cluster_spectra, axis=0)
            cluster_centroids.append(centroid)
            cluster_info.append({
                'id': i,
                'size': len(cluster_spectra),
                'structures': cluster_structures.tolist()
            })

    cluster_centroids = np.array(cluster_centroids)

    # Calculate distances between clusters
    if len(cluster_centroids) > 1:
        cluster_distances = pdist(cluster_centroids, metric='euclidean')
        cluster_linkage = linkage(cluster_distances, method='ward')

        plt.figure(figsize=(12, 8))

        # Create cluster labels with numbers only

        cluster_labels_for_dend = [str(info["id"]) for info in cluster_info]

        dend = dendrogram(cluster_linkage,
                          labels=cluster_labels_for_dend,
                          leaf_rotation=90,
                          leaf_font_size=10)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/dendrogram_cluster_level.png', dpi=300, bbox_inches='tight')
        plt.close()

    else:
        # Single cluster case
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, f'Single Cluster\n\nCluster 1: {cluster_info[0]["size"]} structures',
                 ha='center', va='center', transform=plt.gca().transAxes,
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.savefig(f'{output_dir}/dendrogram_cluster_level.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_detailed_cluster_dendrograms(spectrum_data, cluster_labels, structure_names, n_clusters, output_dir):
    """
    Create detailed dendrograms for each cluster with re-clustering
    """
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Load onset edge data
    onset_data = load_onset_edge_data()

    for cluster_id in tqdm(list(range(1, n_clusters + 1))):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_structures = structure_names[cluster_mask].values
        cluster_spectra = spectrum_data[cluster_mask]

        if len(cluster_indices) > 1:
            # Perform hierarchical clustering within cluster
            cluster_distances = pdist(cluster_spectra, metric='euclidean')
            cluster_linkage = linkage(cluster_distances, method='ward')

            # Adjust dendrogram size
            fig_width = max(8, len(cluster_structures) * 0.8)
            fig_width = min(fig_width, 20)

            plt.figure(figsize=(fig_width, 14))

            # Create labels with material name and properties using onset energy and integral
            enhanced_labels = []
            for name in cluster_structures:
                if name in onset_data:
                    data = onset_data[name]
                    onset_energy = data['onset_energy']
                    absorption_integral = data['absorption_edge_integral']
                    composition = name.split("_")[1] if "_" in name else name
                    label = f"{subscript_numbers(composition)} ($\\omega_{{\\rm o}}$={onset_energy:.1f} eV, $I$={absorption_integral:.1e})"
                else:
                    composition = name.split("_")[1] if "_" in name else name
                    label = f"{subscript_numbers(composition)}\nNo data"
                enhanced_labels.append(label)

            dend = dendrogram(cluster_linkage,
                              labels=enhanced_labels,
                              leaf_rotation=90,
                              leaf_font_size=15)

            plt.tight_layout()
            plt.savefig(f'{output_dir}/dendrogram_cluster_{cluster_id}_detailed.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

        elif len(cluster_indices) == 1:
            # Single structure case
            plt.figure(figsize=(8, 4))
            structure_name = cluster_structures[0]
            if structure_name in onset_data:
                data = onset_data[structure_name]
                info_text = f'Cluster {cluster_id}\n\nSingle structure:\n{structure_name}\n$\\omega_{{\\rm o}}$={data["onset_energy"]:.2f} eV\n$I$={data["absorption_edge_integral"]:.2e}'
            else:
                info_text = f'Cluster {cluster_id}\n\nSingle structure:\n{structure_name}\nNo onset data'

            plt.text(0.5, 0.5, info_text,
                     ha='center', va='center', transform=plt.gca().transAxes,
                     fontsize=12, family='monospace')
            plt.axis('off')
            plt.savefig(f'{output_dir}/dendrogram_cluster_{cluster_id}_detailed.png',
                        dpi=300, bbox_inches='tight')
            plt.close()


def create_onset_integral_scatter_plots(feature_file, output_dir, cluster_labels, n_clusters, n_col, n_row):
    """
    Create scatter plots of onset_energy vs absorption_edge_integral for each cluster
    """

    # Load onset edge data
    onset_data = load_onset_edge_data()

    df = pd.read_json(feature_file)
    structure_names = df.iloc[:, 0]

    # Prepare data for all structures
    all_onset_energies = []
    all_absorption_integrals = []
    all_structure_names = []

    missing_count = 0
    for structure in structure_names:
        if structure in onset_data:
            data = onset_data[structure]
            all_onset_energies.append(data['onset_energy'])
            all_absorption_integrals.append(data['absorption_edge_integral'])
            all_structure_names.append(structure)
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"Warning: {missing_count} structures missing from onset_edge data")

    all_onset_energies = np.array(all_onset_energies)
    all_absorption_integrals = np.array(all_absorption_integrals)

    # Create subplot grid
    subplot_size = 4
    fig_width = n_col * subplot_size
    fig_height = n_row * subplot_size * aspect_all

    fig, axes = plt.subplots(n_row, n_col, figsize=(fig_width, fig_height))

    # Flatten axes array
    if n_row == 1 and n_col == 1:
        axes = [axes]
    elif n_row == 1 or n_col == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Create scatter plots for each cluster
    for cluster_idx in range(1, n_clusters + 1):
        subplot_idx = cluster_idx - 1

        if subplot_idx < len(axes):
            cluster_mask = cluster_labels == cluster_idx
            cluster_structures = structure_names[cluster_mask].values

            # Get data for this cluster
            cluster_onset_energies = []
            cluster_absorption_integrals = []

            for structure in cluster_structures:
                if structure in onset_data:
                    data = onset_data[structure]
                    cluster_onset_energies.append(data['onset_energy'])
                    cluster_absorption_integrals.append(data['absorption_edge_integral'])

            cluster_onset_energies = np.array(cluster_onset_energies)
            cluster_absorption_integrals = np.array(cluster_absorption_integrals)

            # Plot all data in background
            axes[subplot_idx].scatter(all_onset_energies, all_absorption_integrals,
                                      c='lightgray', s=15, zorder=1)

            # Plot cluster data
            if len(cluster_onset_energies) > 0:
                base_color = colors[(cluster_idx - 1) % len(colors)]
                axes[subplot_idx].scatter(cluster_onset_energies, cluster_absorption_integrals,
                                          color=base_color, s=15, zorder=2,
                                          edgecolors='black')

            # Add cluster number and sample count
            n_samples = len(cluster_structures)
            n_with_data = len(cluster_onset_energies)
            if n_samples != n_with_data:
                raise ValueError("strange")
            # axes[subplot_idx].set_title(f'# {cluster_idx} (n={n_samples})\n $\\bar{{onset}} = {np.mean(cluster_onset_energies):.1f} [eV], $\\bar{{int.}} = {np.mean(cluster_absorption_integrals):.1e}', fontsize=18)
            mean_onset = np.mean(cluster_onset_energies)
            mean_log10_integral = np.mean([np.log10(i) for i in cluster_absorption_integrals])
            # title = f'# {cluster_idx} (n={n_samples}, $\\omega_{{\\rm o}}$:{mean_onset:.1f}eV, $\\log(I)$:{mean_log10_integral:.1f})'
            title = (
                f'# {cluster_idx} (n={n_samples}, '
                f'$\\overline{{\\omega_{{\\rm o}}}}$:{mean_onset:.1f}eV, '
                f'$\\overline{{\\log(I)}}$:{mean_log10_integral:.1f})'
            )
            axes[subplot_idx].set_title(title, fontsize=16)
            # axes[subplot_idx].set_xlabel('Onset Energy (eV)', fontsize=14)
            # axes[subplot_idx].set_ylabel('$I$', fontsize=14)
            axes[subplot_idx].set_yscale("log")
            # axes[subplot_idx].set_ylim(10, None)
            axes[subplot_idx].grid(True)
            axes[subplot_idx].tick_params(labelsize=14)

    # Hide unused subplots
    for i in range(n_clusters, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Save scatter plot
    plt.savefig(f'{output_dir}/onset_integral_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    return all_onset_energies, all_absorption_integrals


def save_enhanced_cluster_info(feature_file, output_dir, cluster_labels, n_clusters):
    """
    Save detailed cluster information including onset energy and integral data
    """
    # Load onset edge data
    onset_data = load_onset_edge_data()

    df = pd.read_json(feature_file)
    structure_names = df.iloc[:, 0]

    # Save information for each cluster
    for i in range(1, n_clusters + 1):
        cluster_mask = cluster_labels == i
        cluster_structures = structure_names[cluster_mask].values

        # Get onset and integral information
        min_gaps = []
        onset_energies = []
        direct_gaps = []
        absorption_integrals = []
        valid_structures = []

        for structure in cluster_structures:
            if structure in onset_data:
                data = onset_data[structure]
                min_gaps.append(data['min_gap'])
                onset_energies.append(data['onset_energy'])
                direct_gaps.append(data['direct_gap'])
                absorption_integrals.append(data['absorption_edge_integral'])
                valid_structures.append(structure)

        min_gaps = np.array(min_gaps)
        onset_energies = np.array(onset_energies)
        direct_gaps = np.array(direct_gaps)
        absorption_integrals = np.array(absorption_integrals)

        # Calculate statistics
        with open(f'{output_dir}/cluster_{i}_structures.txt', 'w') as f:
            f.write(f"Cluster {i} ({len(cluster_structures)} structures, {len(valid_structures)} with data):\n")
            f.write("=" * 60 + "\n")

            if len(valid_structures) > 0:
                f.write(f"\nOnset Energy Statistics:\n")
                f.write(f"  Mean: {np.mean(onset_energies):.4f} eV\n")
                f.write(f"  Std:  {np.std(onset_energies):.4f} eV\n")
                f.write(f"  Min:  {np.min(onset_energies):.4f} eV\n")
                f.write(f"  Max:  {np.max(onset_energies):.4f} eV\n")

                f.write(f"\nAbsorption Edge Integral Statistics:\n")
                f.write(f"  Mean: {np.mean(absorption_integrals):.4e}\n")
                f.write(f"  Std:  {np.std(absorption_integrals):.4e}\n")
                f.write(f"  Min:  {np.min(absorption_integrals):.4e}\n")
                f.write(f"  Max:  {np.max(absorption_integrals):.4e}\n")

                f.write(f"\nMin Gap Statistics:\n")
                f.write(f"  Mean: {np.mean(min_gaps):.4f} eV\n")
                f.write(f"  Std:  {np.std(min_gaps):.4f} eV\n")
                f.write(f"  Min:  {np.min(min_gaps):.4f} eV\n")
                f.write(f"  Max:  {np.max(min_gaps):.4f} eV\n")

                f.write(f"\nDirect Gap Statistics (for reference):\n")
                f.write(f"  Mean: {np.mean(direct_gaps):.4f} eV\n")
                f.write(f"  Std:  {np.std(direct_gaps):.4f} eV\n")
                f.write(f"  Min:  {np.min(direct_gaps):.4f} eV\n")
                f.write(f"  Max:  {np.max(direct_gaps):.4f} eV\n")
            else:
                f.write(f"\nNo onset edge data available for structures in this cluster.\n")

            f.write(f"\nStructure List:\n")
            f.write("-" * 40 + "\n")

            for j, structure in enumerate(cluster_structures):
                f.write(f"{j + 1:3d}. {structure}")
                if structure in onset_data:
                    data = onset_data[structure]
                    f.write(
                        f" (onset: {data['onset_energy']:.3f} eV, integral: {data['absorption_edge_integral']:.3e})")
                else:
                    f.write(f" (no onset edge data)")
                f.write("\n")


def hierarchical_clustering_with_overlapped_plots(feature_file, n_clusters, extract_layer, output_dir, n_col, n_row):
    """
    Load pre-computed clustering results and visualize them
    in n_col x n_row subplots with overlapped spectra for each cluster
    """

    # Load data
    df = pd.read_json(feature_file)
    structure_names = df.iloc[:, 0]
    with open(f"{output_dir}/structure_names.json", "w") as fw:
        json.dump(structure_names.values.tolist(), fw, indent=4)
    spectrum_data = np.array(df['reference'].tolist())

    # Energy axis (0-15eV, 1501 points)
    energy = np.linspace(0, 15, spectrum_data.shape[1])

    print(f"Number of data points: {len(df)}")
    print(f"Spectrum data shape: {spectrum_data.shape}")

    # Load pre-computed clustering data
    print("Loading pre-computed clustering data...")
    linkage_matrix = load_clustering_data(extract_layer)

    # Create clusters with specified number using pre-computed linkage matrix
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    print(f"Clustering labels created from pre-computed data. Data classified into {n_clusters} clusters.")

    # Create dendrograms
    print("Creating dendrograms...")
    create_condensed_dendrogram(linkage_matrix, cluster_labels, structure_names, n_clusters, output_dir)
    create_detailed_cluster_dendrograms(spectrum_data, cluster_labels, structure_names, n_clusters, output_dir)

    # Display cluster information
    for i in range(1, n_clusters + 1):
        cluster_mask = cluster_labels == i
        cluster_size = np.sum(cluster_mask)
        print(f"Cluster {i}: {cluster_size} spectra")

    # Create n_col x n_row subplots
    subplot_size = 4
    fig_width = n_col * subplot_size
    fig_height = n_row * subplot_size * aspect_all

    fig, axes = plt.subplots(n_row, n_col, figsize=(fig_width, fig_height))

    # Flatten axes array
    if n_row == 1 and n_col == 1:
        axes = [axes]
    elif n_row == 1 or n_col == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Plot each cluster in subplots
    for cluster_idx in range(1, n_clusters + 1):
        subplot_idx = cluster_idx - 1

        cluster_mask = cluster_labels == cluster_idx
        ids_in_cluster = np.where(cluster_labels == cluster_idx)[0]
        print(f"Cluster {cluster_idx}: indices {ids_in_cluster}")
        print(f"Structures: {structure_names[cluster_mask].values}")

        if subplot_idx < len(axes):
            cluster_mask = cluster_labels == cluster_idx
            cluster_spectra = spectrum_data[cluster_mask]
            cluster_names = structure_names[cluster_mask].values

            n_spectra = len(cluster_spectra)

            if n_spectra > 0:
                # Plot all spectra in cluster
                base_color = colors[(cluster_idx - 1) % len(colors)]

                # Plot individual spectra
                for spectrum in cluster_spectra:
                    axes[subplot_idx].plot(energy, spectrum,
                                           color=base_color,
                                           linewidth=1.2)

                # Plot mean spectrum
                mean_spectrum = np.mean(cluster_spectra, axis=0)
                axes[subplot_idx].plot(energy, mean_spectrum,
                                       color="black",
                                       linewidth=2.5,
                                       label=f'Mean (n={n_spectra})')

                # Add cluster number and sample count in bottom right
                axes[subplot_idx].text(0.98, 0.02, f'# {cluster_idx}\n(n={n_spectra})',
                                       transform=axes[subplot_idx].transAxes,
                                       fontsize=20, fontweight='bold',
                                       ha='right', va='bottom',
                                       bbox=dict(boxstyle='round', facecolor='white'))

                axes[subplot_idx].grid(True)
                axes[subplot_idx].tick_params(labelsize=16)
                axes[subplot_idx].set_ylim(2.01, 6.3)
            else:
                # Empty cluster case
                axes[subplot_idx].text(0.5, 0.5, f'Cluster {cluster_idx}\n(Empty)',
                                       ha='center', va='center',
                                       transform=axes[subplot_idx].transAxes,
                                       fontsize=12)

    # Hide unused subplots
    for i in range(n_clusters, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Save main plot
    plt.savefig(f'{output_dir}/all_clusters_overlapped.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Create comparison plot of all cluster mean spectra
    plt.figure(figsize=(12, 8))

    for i in range(1, n_clusters + 1):
        cluster_mask = cluster_labels == i
        cluster_spectra = spectrum_data[cluster_mask]

        if len(cluster_spectra) > 0:
            mean_spectrum = np.mean(cluster_spectra, axis=0)
            base_color = colors[(i - 1) % len(colors)]

            plt.plot(energy, mean_spectrum,
                     color=base_color,
                     linewidth=2.5,
                     label=f'Cluster {i} (n={np.sum(cluster_mask)})')

    plt.legend()
    plt.savefig(f'{output_dir}/clusters_mean_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save clustering results to CSV
    result_df = pd.DataFrame({
        'structure_file': structure_names,
        'cluster': cluster_labels
    })
    result_df.to_csv(f'{output_dir}/clustering_results.csv', index=False)

    print(f"Results saved to '{output_dir}' directory:")
    print(f"- dendrogram_cluster_level.png: Cluster-level dendrogram")
    print(f"- dendrogram_cluster_i_detailed.png: Detailed dendrogram for each cluster")
    print(f"- all_clusters_overlapped.png: Overlapped spectra plot ({n_col}x{n_row})")
    print(f"- clusters_mean_comparison.png: Mean spectra comparison for all clusters")
    print(f"- cluster_i_structures.txt: Structure files and statistics for each cluster")
    print(f"- clustering_results.csv: Clustering results")

    # Create onset vs integral scatter plots
    print("Creating onset vs integral scatter plots...")
    create_onset_integral_scatter_plots(feature_file, output_dir, cluster_labels, n_clusters, n_col, n_row)
    save_enhanced_cluster_info(feature_file, output_dir, cluster_labels, n_clusters)
    print(f"- onset_integral_scatter.png: Onset Energy vs Absorption Edge Integral scatter plot")

    # Create individual plots for each cluster with legends
    print("Creating individual cluster plots with legends...")
    for cluster_idx in list(tqdm(range(1, n_clusters + 1))):
        cluster_mask = cluster_labels == cluster_idx
        cluster_spectra = spectrum_data[cluster_mask]
        cluster_names = structure_names[cluster_mask].values

        if len(cluster_spectra) > 0:
            plt.figure(figsize=(12, 12))

            # Plot individual spectra with material names as legends
            for i, (spectrum, name) in enumerate(zip(cluster_spectra, cluster_names)):
                # Extract composition from structure name
                composition = name.split("_")[1] if "_" in name else name

                plt.plot(energy, spectrum,
                         linewidth=2.0,
                         label=subscript_numbers(composition))

            plt.title(f'Cluster # {cluster_idx} (n={len(cluster_spectra)})', fontsize=20)
            plt.grid(True)
            max_row = 25
            n_col = (len(cluster_names) + (max_row - 1)) // max_row
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20, ncol=n_col)
            plt.tick_params(labelsize=15)
            plt.xlim(0, 15)
            plt.ylim(2.01, 6.3)

            # Save individual cluster plot
            plt.savefig(f'{output_dir}/cluster_{cluster_idx}_individual.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

    print(f"- cluster_i_individual.png: Individual plots for each cluster with material legends")

    return cluster_labels, linkage_matrix


def main():
    # Configuration
    feature_file = "features/features.json"
    extract_layer = sys.argv[1]
    n_col = int(sys.argv[2])
    n_row = int(sys.argv[3])
    n_clusters = n_col * n_row
    output_dir = sys.argv[4] if len(sys.argv) > 5 else f"cluster/{extract_layer}/{n_col}x{n_row}_plot"
    os.makedirs(output_dir, exist_ok=True)

    # Validate that n_clusters matches n_col * n_row
    if n_clusters != n_col * n_row:
        print(
            f"Error: Number of clusters ({n_clusters}) must equal n_col * n_row ({n_col} * {n_row} = {n_col * n_row})")
        sys.exit(1)

    print(f"Feature file: {feature_file}")
    print(f"Extract layer: {extract_layer}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Subplot grid: {n_col}x{n_row}")
    print(f"Output directory: {output_dir}")

    # Execute clustering visualization
    cluster_labels, linkage_matrix = hierarchical_clustering_with_overlapped_plots(
        feature_file, n_clusters, extract_layer, output_dir, n_col, n_row
    )

    print("Clustering visualization completed.")


# Main execution
if __name__ == "__main__":
    main()