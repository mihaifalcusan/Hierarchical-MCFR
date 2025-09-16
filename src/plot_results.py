import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def parse_and_load_results(results_dir='results'):
    """
    Walks through the results directory, parses experiment parameters, and loads results.
    """
    all_results = []
    
    for root, _, files in os.walk(results_dir):
        if 'cate_results.npz' in files:
            try:
                parts = root.replace('\\', '/').split('/')
                scenario, model = parts[-4], parts[-3]
                params = parts[-2].split('_')
                n_samples, kappa = int(params[0].replace('n', '')), float(params[1].replace('k', ''))
                seed = int(parts[-1].replace('seed', ''))
                
                result_data = np.load(os.path.join(root, 'cate_results.npz'))
                pehe_scores = result_data['pehe']
                
                all_results.append({
                    'scenario': scenario, 'model': model, 'n_samples': n_samples,
                    'kappa': kappa, 'seed': seed, 'pehe_t1_vs_t0': pehe_scores[0],
                    'pehe_t2_vs_t0': pehe_scores[1], 'pehe_t3_vs_t0': pehe_scores[2]
                })
            except Exception as e:
                print(f"Skipping directory {root} due to parsing/loading error: {e}")

    if not all_results:
        return None
        
    return pd.DataFrame(all_results)

def plot_pehe_vs_kappa(df):
    """
    Generates a faceted point plot of Mean PEHE vs. Kappa for all models.
    """
    if df is None or df.empty:
        print("DataFrame is empty. Cannot generate PEHE vs. Kappa plot.")
        return

    print("Generating PEHE vs. Kappa summary plot...")
    
    # Filter out any failed models for a cleaner plot
    df_plot = df[~df['model'].str.contains('structured')].copy()

    df_long = pd.melt(df_plot, 
                      id_vars=['scenario', 'model', 'kappa'],
                      value_vars=['pehe_t1_vs_t0', 'pehe_t2_vs_t0', 'pehe_t3_vs_t0'],
                      var_name='effect', value_name='pehe')
    df_long['effect'] = df_long['effect'].str.replace('pehe_', '').str.upper()

    g = sns.catplot(data=df_long, x='kappa', y='pehe', hue='model',
                    col='effect', row='scenario', kind='point',
                    dodge=True, errorbar=('ci', 95), capsize=.1,
                    height=4, aspect=1.2, sharey=False)

    g.fig.suptitle('Model Performance (PEHE) vs. Treatment Overlap (Kappa)', y=1.03, fontsize=16)
    g.set_axis_labels('Overlap Parameter (Kappa)', 'Mean PEHE (Lower is Better)')
    g.set_titles(row_template="{row_name} Scenario", col_template="{col_name}")
    g.despine(left=True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plt.savefig('summary_pehe_vs_kappa.png', dpi=300)
    print("Plot saved as 'summary_pehe_vs_kappa.png'")
    plt.show()

def plot_true_vs_est_cate(results_path):
    """
    Creates a scatter plot of true vs. estimated CATE for a single experiment run.
    """
    print(f"\nGenerating True vs. Estimated CATE plot for: {results_path}")
    try:
        data = np.load(results_path)
        est_cate, true_cate = data['est_cate'], data['true_cate']
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_path}")
        return

    cate_t1_true, cate_t1_est = true_cate[:, 0], est_cate[:, 0]
    min_val = min(cate_t1_true.min(), cate_t1_est.min())
    max_val = max(cate_t1_true.max(), cate_t1_est.max())

    plt.figure(figsize=(7, 7))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Calibration')
    sns.scatterplot(x=cate_t1_true, y=cate_t1_est, alpha=0.5, s=50, label='Individual Estimates')
    plt.xlabel("True CATE (T1 vs T0)"), plt.ylabel("Estimated CATE (T1 vs T0)")
    plt.title("True vs. Estimated CATE Scatter Plot", fontsize=14)
    plt.legend(), plt.grid(True), plt.axis('square'), plt.tight_layout()
    plt.savefig('cate_scatter_plot.png', dpi=300)
    print("Plot saved as 'cate_scatter_plot.png'")
    plt.show()

def plot_cate_vs_feature(results_path, feature_index=0, feature_name='Feature'):
    """
    Plots true and estimated CATE against a key feature to check for heterogeneity.
    """
    print(f"\nGenerating CATE vs. Feature plot for: {results_path}")
    try:
        data = np.load(results_path)
        est_cate, true_cate, X_test = data['est_cate'], data['true_cate'], data['X_test']
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: Could not load required data from {results_path}. Did you save X_test? Error: {e}")
        return

    cate_t1_true, cate_t1_est = true_cate[:, 0], est_cate[:, 0]
    feature_values = X_test[:, feature_index]

    plt.figure(figsize=(10, 6))
    # Plotting a sample to avoid overplotting
    sample_indices = np.random.choice(len(feature_values), size=500, replace=False)
    sns.scatterplot(x=feature_values[sample_indices], y=cate_t1_true[sample_indices], alpha=0.5, label='True CATE')
    sns.scatterplot(x=feature_values[sample_indices], y=cate_t1_est[sample_indices], alpha=0.5, label='Estimated CATE')
    plt.xlabel(f"{feature_name} (X_{feature_index+1})"), plt.ylabel("CATE (T1 vs T0)")
    plt.title(f"CATE vs. {feature_name}", fontsize=14)
    plt.legend(), plt.grid(True), plt.tight_layout()
    plt.savefig('cate_vs_feature_plot.png', dpi=300)
    print("Plot saved as 'cate_vs_feature_plot.png'")
    plt.show()


if __name__ == '__main__':
    # --- Plot 1: Overall Summary ---
    # This aggregates all experiment runs
    results_df = parse_and_load_results()
    plot_pehe_vs_kappa(results_df)

    # --- Plots 2 & 3: Deep Dive into a SINGLE Run ---
    # We'll pick one of the best-performing models to visualize in detail.
    # For example, the hierarchical model on the medication data.
    EXAMPLE_RUN_PATH = "results/medication/hierarchical_mcfr/n5000_k2.0/seed1/cate_results.npz"
    
    plot_true_vs_est_cate(EXAMPLE_RUN_PATH)
    plot_cate_vs_feature(EXAMPLE_RUN_PATH, feature_index=0, feature_name='Disease Severity')