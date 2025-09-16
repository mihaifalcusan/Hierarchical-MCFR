import os
import numpy as np
import pandas as pd
from scipy import stats

def parse_and_load_results(results_dir='results'):
    """
    Walks through the results directory, parses experiment parameters from
    directory names, and loads the PEHE results.
    """
    all_results = []
    
    for root, _, files in os.walk(results_dir):
        if 'cate_results.npz' in files:
            # --- Parse parameters from the directory path ---
            try:
                parts = root.replace('\\', '/').split('/')
                scenario = parts[-4]
                model = parts[-3]
                params = parts[-2].split('_')
                n_samples = int(params[0].replace('n', ''))
                kappa = float(params[1].replace('k', ''))
                seed = int(parts[-1].replace('seed', ''))
            except (IndexError, ValueError) as e:
                print(f"Could not parse directory name: {root}. Skipping.")
                continue

            # --- Load the PEHE scores ---
            try:
                result_data = np.load(os.path.join(root, 'cate_results.npz'))
                pehe_scores = result_data['pehe']
                
                result_entry = {
                    'scenario': scenario,
                    'model': model,
                    'n_samples': n_samples,
                    'kappa': kappa,
                    'seed': seed,
                    'pehe_t1_vs_t0': pehe_scores[0],
                    'pehe_t2_vs_t0': pehe_scores[1],
                    'pehe_t3_vs_t0': pehe_scores[2]
                }
                all_results.append(result_entry)
            except Exception as e:
                print(f"Could not load or read file in {root}: {e}. Skipping.")

    if not all_results:
        print("No result files found. Please run the experiments first.")
        return None
        
    return pd.DataFrame(all_results)


def calculate_summary_stats(df):
    """
    Groups the results by condition and calculates mean and 95% CI for PEHE.
    """
    if df is None:
        return None
        
    # Define the columns we want to aggregate
    pehe_cols = ['pehe_t1_vs_t0', 'pehe_t2_vs_t0', 'pehe_t3_vs_t0']
    
    # Define aggregation functions: mean and a function to get the 95% CI
    def get_95_ci(data):
        if len(data) < 2:
            return 0
        mean, sem = np.mean(data), stats.sem(data)
        return sem * stats.t.ppf((1 + 0.95) / 2., len(data)-1)

    aggregations = {col: ['mean', get_95_ci] for col in pehe_cols}
    
    # Group by experimental conditions
    summary = df.groupby(['scenario', 'model', 'n_samples', 'kappa']).agg(aggregations).reset_index()
    
    # Clean up column names
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    
    return summary


def main():
    """Main function to run the analysis and print results."""
    
    results_df = parse_and_load_results()
    
    if results_df is not None:
        summary_df = calculate_summary_stats(results_df)
        
        print("\n--- Aggregated Experiment Results ---")
        
        for _, row in summary_df.iterrows():
            print("-" * 60)
            print(f"Scenario: {row['scenario']}, Model: {row['model']}, N: {row['n_samples']}, Kappa: {row['kappa']}")
            
            # Print PEHE for T1 vs T0
            mean_t1 = row['pehe_t1_vs_t0_mean']
            ci_t1 = row['pehe_t1_vs_t0_get_95_ci']
            print(f"  PEHE (T1 vs T0): {mean_t1:.4f} ± {ci_t1:.4f}")

            # Print PEHE for T2 vs T0
            mean_t2 = row['pehe_t2_vs_t0_mean']
            ci_t2 = row['pehe_t2_vs_t0_get_95_ci']
            print(f"  PEHE (T2 vs T0): {mean_t2:.4f} ± {ci_t2:.4f}")

            # Print PEHE for T3 vs T0
            mean_t3 = row['pehe_t3_vs_t0_mean']
            ci_t3 = row['pehe_t3_vs_t0_get_95_ci']
            print(f"  PEHE (T3 vs T0): {mean_t3:.4f} ± {ci_t3:.4f}")
        
        print("-" * 60)


if __name__ == '__main__':
    main()