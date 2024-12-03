import os
import pandas as pd
import argparse

def gather_return(env_id, idx):
    base_path = f'./data/{env_id}'
    methods = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    print(f"==> Loading [mode:{idx}] returns of methods: ", ', '.join(methods))
    combined_df = pd.DataFrame()
    for method in methods:
        file_path = os.path.join(base_path, method, str(idx), 'returns.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df.columns = [f'{method}-{idx}-{col}' if col == 'episodic_return' else col for col in df.columns]
            df = df.drop_duplicates(subset=['global_step'])
            if combined_df.empty:
                combined_df = df
            else:
                combined_df = combined_df.merge(df, on='global_step', how='outer')
        else:
            print(f"====> File not found for [method-{method}|mode-{idx}] : ", file_path)
            quit(1)
    
    combined_df.to_csv(f'./data/{env_id}/task_{idx}.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gather rewards for a specific environment.')
    parser.add_argument("--env", type=str, choices=["SpaceInvaders", "Freeway"], default="Freeway")
    args = parser.parse_args()
    
    num_tasks = (8 if args.env == 'Freeway' else 10)
    for idx in range(0, num_tasks):
        gather_return(args.env, idx)