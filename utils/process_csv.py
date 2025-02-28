import os
import pandas as pd

def process_csv_files(folder_path):
    # Initialize an empty list to store the results
    results = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Extract the last row (average results)
            avg_results = df.iloc[-1]

            # Extract the classifier and benchmark from the filename
            parts = filename.split('_')
            benchmark = parts[1]
            classifier = '_'.join(parts[2:]).replace('.csv', '')

            # Append the results to the list
            results.append({
                'Benchmark': benchmark,
                'Classifier': classifier,
                'TP': avg_results['TP'],
                'FP': avg_results['FP'],
                'FN': avg_results['FN'],
                'T': avg_results['T'],
                'T_T': avg_results['T_T'],
                'T_P': avg_results['T_P'],
                'Instances': avg_results['Instances'],
                'Precision': avg_results['Precision'],
                'Recall': avg_results['Recall']
            })

    # Create a dataframe from the results
    results_df = pd.DataFrame(results)

    return results_df


import os
import pandas as pd


def process_csv_files(folder_path, benchmarks, classifiers):
    # Initialize an empty list to store the results
    results = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Extract the last row (average results)
            avg_results = df.iloc[-1]

            # Identify the benchmark and classifier from the filename
            for benchmark in benchmarks:
                if benchmark in filename:
                    for classifier in classifiers:
                        if classifier in filename:
                            # Append the results to the list
                            results.append({
                                'Benchmark': benchmark,
                                'Classifier': classifier,
                                'TP': avg_results['TP'],
                                'FP': avg_results['FP'],
                                'FN': avg_results['FN'],
                                'T': avg_results['T'],
                                'T_T': avg_results['T_T'],
                                'T_P': avg_results['T_P'],
                                'Instances': avg_results['Instances'],
                                'Precision': avg_results['Precision'],
                                'Recall': avg_results['Recall']
                            })
                            break
                    break

    # Create a dataframe from the results
    results_df = pd.DataFrame(results)

    return results_df


import pandas as pd


def transform_results_df(results_df):
    # Get the unique classifiers and benchmarks
    classifiers = results_df['Classifier'].unique()
    benchmarks = results_df['Benchmark'].unique()

    # Create a multi-level column index
    columns = pd.MultiIndex.from_product([['Precision', 'Recall'], benchmarks], names=['Metric', 'Benchmark'])

    # Initialize an empty DataFrame with classifiers as the index and the multi-level columns
    transformed_df = pd.DataFrame(index=classifiers, columns=columns)

    # Populate the DataFrame with precision and recall values
    for classifier in classifiers:
        for benchmark in benchmarks:
            # Filter the results for the current benchmark and classifier
            filtered_results = results_df[
                (results_df['Benchmark'] == benchmark) & (results_df['Classifier'] == classifier)]

            if not filtered_results.empty:
                precision = filtered_results['Precision'].values[0]
                recall = filtered_results['Recall'].values[0]

                # Set the precision and recall values in the transformed DataFrame
                transformed_df.at[classifier, ('Precision', benchmark)] = precision
                transformed_df.at[classifier, ('Recall', benchmark)] = recall

    return transformed_df

# Example usage
all_benchmarks = ["sudoku", "golomb", "exam_timetabling", "nurse_rostering"]
all_classifiers = ["random_forest", "MLP", "CategoricalNB", "DT", "KNN", "CN2", "countcp"]
folder_path = '../results/no_noise/'
results_df = process_csv_files(folder_path,all_benchmarks, all_classifiers)
print("results: ", results_df)

transformed_df = transform_results_df(results_df)
print("transformed_df: ", transformed_df)

# Save the transformed DataFrame to a CSV file
transformed_df.to_csv(f'{folder_path}transformed_results.csv')