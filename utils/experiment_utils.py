import re
import json
import os
import pickle
import csv
import shutil

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import seaborn as sns
from utils.generic_utils import get_variables_from_constraints, get_var_dims
from ProblemInstance import ProblemInstance


def load_instances(benchmark_name):
    # Define the base directory where all benchmarks are stored
    base_directory = 'benchmarks/'  # Replace with the actual path

    # Construct the full path to the benchmark directory
    benchmark_directory = os.path.join(base_directory, benchmark_name)

    # Dictionary to store the instances with their names as keys
    instances = []

    # Check if the benchmark directory exists
    if not os.path.exists(benchmark_directory):
        raise NotImplementedError('Benchmark given not implemented')

    # Iterate over all files in the benchmark directory
    for idx, filename in enumerate(os.listdir(benchmark_directory)):
        # Check if the file is a pickle file
        if filename.endswith('.pickle'):

            # Extract the base name of the instance (without the extension)
            instance_name = filename[:-7]
            # Construct the full path to the pickle file
            pickle_path = os.path.join(benchmark_directory, filename)
            # Construct the full path to the corresponding json file
            json_path = os.path.join(benchmark_directory, f"{instance_name}.json")

            # Check if the corresponding json file exists
            if os.path.exists(json_path):

                # Load the pickle file
                with open(pickle_path, 'rb') as f:
                    csp = pickle.load(f)
                # Load the json file
                with open(json_path, 'r') as f:
                    params = json.load(f)

                par_path = os.path.join(benchmark_directory, f"{instance_name}.par")
                if os.path.exists(par_path):
                    # Load the partitions file
                    with open(par_path, 'rb') as f:
                        par = pickle.load(f)
                else:
                    par = []

                # Store the loaded data in the dictionary
                instances.append({'name': instance_name, 'csp': list(csp), 'params': params, 'partitions': par})

            else:
                raise IOError(f"Corresponding json file for '{filename}' not found.")

    return instances


def get_benchmark_models(benchmark):
    # Get all instances for the benchmark using load_instances
    instances = load_instances(benchmark)
    
    # Sort instances by some complexity measure (e.g. number in filename)
    instances.sort(key=lambda x: int(re.findall(r'\d+', x['name'])[-1]))
    
    # Convert instances into ProblemInstance objects
    problem_instances = []
    for instance in instances:
        problem_instance = ProblemInstance(
            constraints=instance['csp'],
            params=instance['params'],
            name=f"{benchmark}_instance",
            custom_partitions=[instance['partitions']] if instance['partitions'] else []
        )
        problem_instances.append(problem_instance)

    return problem_instances


def construct_classifier(classifier_name):
    classifier = None

    if classifier_name == "random_forest":
        classifier = RandomForestClassifier(n_estimators=100)
    elif classifier_name == "MLP":
        classifier = MLPClassifier(activation='relu', solver='adam', max_iter=100000, random_state=1,
                                   learning_rate_init=0.001, hidden_layer_sizes=64)
    elif classifier_name == "CategoricalNB":
        classifier = CategoricalNB()
    elif classifier_name == "GaussianNB":
        classifier = GaussianNB()
    elif classifier_name == "SVM":
        classifier = SVC(C=10, gamma='auto', kernel='poly')
    elif classifier_name == "DT":
        classifier = DecisionTreeClassifier(max_leaf_nodes=10)
    elif classifier_name == "KNN":
        classifier = KNeighborsClassifier(weights='uniform', metric='euclidean', n_neighbors=2)
    elif classifier_name == "countcp":
        pass
    elif classifier_name in ["CN2", "countcp"]:
        pass
    else:
        raise Exception("Classifier not implemented in current system")

    return classifier


def save_results(clf, bench, metrics, instance, noise_p=None, noise_mode=None):
    assert bench is not None, "to save the results, we need to have the benchmark name"
    assert clf is not None, "to save the results, we need to have the classifier name"
    assert metrics is not None, "to save the results, we need to have the metrics"
    assert instance is not None, "to save the results, we need to have the instance parameters"

    print("\n\nGeneralized ------------------------")

    print("Total time: ", metrics.total_time)
    print(
        f"noise: {noise_p}, found: {metrics.found_constraints}, true positive: {metrics.true_pos}, false positive: {metrics.false_pos}, false_negative: {metrics.false_neg}")

    res_name = ["results/results"]
    res_name.append(bench)
    res_name.append(f"{instance[0]}_{instance[1]}")

    if noise_p is not None:
        assert noise_mode is not None
        res_name.append(str(noise_p))
        res_name.append(noise_mode)

    res_name.append(str(clf))

    res_name.append(bench)

    results_file = "_".join(res_name)

    file_exists = os.path.isfile(results_file)
    f = open(results_file, "a")

    if not file_exists:
        results = "CL\tTP\tFP\tFN\tT\tT_T\tT_P\n"
    else:
        results = ""

    results += str(metrics.found_constraints) + "\t" + str(metrics.true_pos) + "\t" + str(metrics.false_pos) \
               + "\t" + str(metrics.false_neg) + "\t" + str(metrics.total_time) + "\t" + str(metrics.training_time) \
               + "\t" + str(metrics.prediction_time) + "\n"

    f.write(results)
    f.close()


def parse_last_entry(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        last_entry = lines[-1].strip().split()
        return last_entry


def extract_instances(file_name):
    match = re.search(r'_(\d+)_(\d+)_', file_name)
    if match:
        return match.group(1), match.group(2)
    return None, None


def calculate_precision_recall(tp, fp, fn):
    tp, fp, fn = float(tp), float(fp), float(fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall

def calculate_averages(results):
    num_columns = len(results[0])
    sums = [0] * num_columns
    for result in results:
        for i in range(num_columns):
            if i != 7:  # Skip the 'Instances' column
                sums[i] += float(result[i])
    averages = [s / len(results) for s in sums]
    averages[7] = 'average_results'  # Set the 'Instances' column to 'average_results'
    return averages


def process_files(folder_path, benchmark, classifier, output_csv=None):
    if output_csv is None:
        output_csv = f"{folder_path}results_{benchmark}_{classifier}.csv"
        print(f"processing folder {folder_path}, benchmark {benchmark} and classifier {classifier}")
    results = []
    for file_name in os.listdir(folder_path):
        if file_name.startswith(f"results_{benchmark}") and file_name.endswith(f"{classifier}_{benchmark}"):
            file_path = os.path.join(folder_path, file_name)
            last_entry = parse_last_entry(file_path)
            instance1, instance2 = extract_instances(file_name)
            if instance1 and instance2:
                tp, fp, fn = last_entry[1], last_entry[2], last_entry[3]
                precision, recall = calculate_precision_recall(tp, fp, fn)
                last_entry.append(f"{instance1}_{instance2}")
                last_entry.append(precision)
                last_entry.append(recall)
                results.append(last_entry)

    averages = calculate_averages(results)

    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['CL', 'TP', 'FP', 'FN', 'T', 'T_T', 'T_P', 'Instances', 'Precision', 'Recall'])
        csvwriter.writerows(results)
        csvwriter.writerow(averages)

def move_files_with_substring(source_dir, target_dir, substring):
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    counter = 0
    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        if substring in filename:
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)
            # Move the file to the target directory
            shutil.move(source_file, target_file)
            print(f"Moved: {filename}")
            counter += 1

    print(f"Moved {counter} files in total")

def parse_noise_results(main_folder):

    # Define the subfolders
    subfolders = ['0', '05', '10', '15', '20']

    # Initialize dictionaries to store the results
    precision_results = {}
    recall_results = {}

    et_precision = {}
    golomb_precision = {}
    nr_precision = {}
    sudoku_precision = {}

    et_recall = {}
    golomb_recall = {}
    nr_recall = {}
    sudoku_recall = {}

    # Process each subfolder
    for subfolder in subfolders:
        # Define the path to the transformed_results.csv file
        file_path = os.path.join(main_folder, subfolder, 'transformed_results.csv')

        # Read the CSV file
        df = pd.read_csv(file_path)
        print(subfolder)
        print(df)
        print(df['Precision.1'])
        # Calculate the average precision and recall for each classifier
        for classifier in df['Metric']:
            if classifier == "Benchmark":
                continue

            if classifier not in precision_results:
                precision_results[classifier] = []
                recall_results[classifier] = []

                et_precision[classifier] = []
                golomb_precision[classifier] = []
                nr_precision[classifier] = []
                sudoku_precision[classifier] = []
                et_recall[classifier] = []
                golomb_recall[classifier] = []
                nr_recall[classifier] = []
                sudoku_recall[classifier] = []

            print("DEBUG")
            print(df['Metric'] == classifier)
            print(df[df['Metric'] == classifier])
            precision_avg = \
            df[df['Metric'] == classifier].filter(like='Precision').apply(pd.to_numeric, errors='coerce').mean(
                axis=1).values[0]
            recall_avg = \
            df[df['Metric'] == classifier].filter(like='Recall').apply(pd.to_numeric, errors='coerce').mean(
                axis=1).values[0]

            precision_et = df[['Metric','Precision']][df['Metric'] == classifier].filter(like='Precision').apply(pd.to_numeric, errors='coerce').mean(axis=1).values[0]
            precision_golomb = df[['Metric','Precision.1']][df['Metric'] == classifier].filter(like='Precision').apply(pd.to_numeric, errors='coerce').mean(axis=1).values[0]
            precision_nr = df[['Metric','Precision.2']][df['Metric'] == classifier].filter(like='Precision').apply(pd.to_numeric, errors='coerce').mean(axis=1).values[0]
            precision_sudoku = df[['Metric','Precision.3']][df['Metric'] == classifier].filter(like='Precision').apply(pd.to_numeric, errors='coerce').mean(axis=1).values[0]

            recall_et = df[['Metric','Recall']][df['Metric'] == classifier].filter(like='Recall').apply(pd.to_numeric, errors='coerce').mean(axis=1).values[0]
            recall_golomb = df[['Metric','Recall.1']][df['Metric'] == classifier].filter(like='Recall').apply(pd.to_numeric, errors='coerce').mean(axis=1).values[0]
            recall_nr = df[['Metric','Recall.2']][df['Metric'] == classifier].filter(like='Recall').apply(pd.to_numeric, errors='coerce').mean(axis=1).values[0]
            recall_sudoku = df[['Metric','Recall.3']][df['Metric'] == classifier].filter(like='Recall').apply(pd.to_numeric, errors='coerce').mean(axis=1).values[0]

            precision_results[classifier].append((int(subfolder), precision_avg))
            recall_results[classifier].append((int(subfolder), recall_avg))

            et_precision[classifier].append((int(subfolder), precision_et))
            golomb_precision[classifier].append((int(subfolder), precision_golomb))
            nr_precision[classifier].append((int(subfolder), precision_nr))
            sudoku_precision[classifier].append((int(subfolder), precision_sudoku))

            et_recall[classifier].append((int(subfolder), recall_et))
            golomb_recall[classifier].append((int(subfolder), recall_golomb))
            nr_recall[classifier].append((int(subfolder), recall_nr))
            sudoku_recall[classifier].append((int(subfolder), recall_sudoku))

    # Create dataframes for average precision and recall results
    precision_df = pd.DataFrame(columns=['noise percentage', 'Metric', 'Precision'])
    recall_df = pd.DataFrame(columns=['noise percentage', 'Metric', 'Recall'])

    # Create dataframes for benchmark precision and recall results
    et_precision_df = pd.DataFrame(columns=['noise percentage', 'Metric', 'Precision'])
    et_recall_df = pd.DataFrame(columns=['noise percentage', 'Metric', 'Recall'])
    g_precision_df = pd.DataFrame(columns=['noise percentage', 'Metric', 'Precision'])
    g_recall_df = pd.DataFrame(columns=['noise percentage', 'Metric', 'Recall'])
    nr_precision_df = pd.DataFrame(columns=['noise percentage', 'Metric', 'Precision'])
    nr_recall_df = pd.DataFrame(columns=['noise percentage', 'Metric', 'Recall'])
    s_precision_df = pd.DataFrame(columns=['noise percentage', 'Metric', 'Precision'])
    s_recall_df = pd.DataFrame(columns=['noise percentage', 'Metric', 'Recall'])

    # Populate the dataframes
    for classifier in precision_results:
        #average
        for noise_percentage, precision_avg in precision_results[classifier]:
            precision_df = pd.concat([precision_df, pd.DataFrame(
                {'noise percentage': [noise_percentage], 'Metric': [classifier], 'Precision': [precision_avg]})],
                                     ignore_index=True)
        for noise_percentage, recall_avg in recall_results[classifier]:
            recall_df = pd.concat([recall_df, pd.DataFrame(
                {'noise percentage': [noise_percentage], 'Metric': [classifier], 'Recall': [recall_avg]})],
                                  ignore_index=True)

        # per benchmark
        for noise_percentage, precision_avg in et_precision[classifier]:
            et_precision_df = pd.concat([et_precision_df, pd.DataFrame(
                {'noise percentage': [noise_percentage], 'Metric': [classifier], 'Precision': [precision_avg]})],
                                     ignore_index=True)
        for noise_percentage, recall_avg in et_recall[classifier]:
            et_recall_df = pd.concat([et_recall_df, pd.DataFrame(
                {'noise percentage': [noise_percentage], 'Metric': [classifier], 'Recall': [recall_avg]})],
                                  ignore_index=True)
        for noise_percentage, precision_avg in golomb_precision[classifier]:
            g_precision_df = pd.concat([g_precision_df, pd.DataFrame(
                {'noise percentage': [noise_percentage], 'Metric': [classifier], 'Precision': [precision_avg]})],
                                     ignore_index=True)
        for noise_percentage, recall_avg in golomb_recall[classifier]:
            g_recall_df = pd.concat([g_recall_df, pd.DataFrame(
                {'noise percentage': [noise_percentage], 'Metric': [classifier], 'Recall': [recall_avg]})],
                                  ignore_index=True)
        for noise_percentage, precision_avg in nr_precision[classifier]:
            nr_precision_df = pd.concat([nr_precision_df, pd.DataFrame(
                {'noise percentage': [noise_percentage], 'Metric': [classifier], 'Precision': [precision_avg]})],
                                     ignore_index=True)
        for noise_percentage, recall_avg in nr_recall[classifier]:
            nr_recall_df = pd.concat([nr_recall_df, pd.DataFrame(
                {'noise percentage': [noise_percentage], 'Metric': [classifier], 'Recall': [recall_avg]})],
                                  ignore_index=True)
        for noise_percentage, precision_avg in sudoku_precision[classifier]:
            s_precision_df = pd.concat([s_precision_df, pd.DataFrame(
                {'noise percentage': [noise_percentage], 'Metric': [classifier], 'Precision': [precision_avg]})],
                                     ignore_index=True)
        for noise_percentage, recall_avg in sudoku_recall[classifier]:
            s_recall_df = pd.concat([s_recall_df, pd.DataFrame(
                {'noise percentage': [noise_percentage], 'Metric': [classifier], 'Recall': [recall_avg]})],
                                  ignore_index=True)

    # Pivot the average dataframes
    precision_pivot = precision_df.pivot(index='Metric', columns='noise percentage', values='Precision').reset_index()
    recall_pivot = recall_df.pivot(index='Metric', columns='noise percentage', values='Recall').reset_index()

    # Pivot the benchmark dataframes
    et_precision_pivot = et_precision_df.pivot(index='Metric', columns='noise percentage', values='Precision').reset_index()
    et_recall_pivot = et_recall_df.pivot(index='Metric', columns='noise percentage', values='Recall').reset_index()
    g_precision_pivot = g_precision_df.pivot(index='Metric', columns='noise percentage',
                                            values='Precision').reset_index()
    g_recall_pivot = g_recall_df.pivot(index='Metric', columns='noise percentage', values='Recall').reset_index()
    nr_precision_pivot = nr_precision_df.pivot(index='Metric', columns='noise percentage',
                                            values='Precision').reset_index()
    nr_recall_pivot = nr_recall_df.pivot(index='Metric', columns='noise percentage', values='Recall').reset_index()
    s_precision_pivot = s_precision_df.pivot(index='Metric', columns='noise percentage',
                                            values='Precision').reset_index()
    s_recall_pivot = s_recall_df.pivot(index='Metric', columns='noise percentage', values='Recall').reset_index()

    # Save the results to CSV files
    precision_pivot.to_csv(os.path.join(main_folder, 'average_precision_results_pivot.csv'), index=False)
    recall_pivot.to_csv(os.path.join(main_folder, 'average_recall_results_pivot.csv'), index=False)

    et_precision_pivot.to_csv(os.path.join(main_folder, 'et_precision_results_pivot.csv'), index=False)
    et_recall_pivot.to_csv(os.path.join(main_folder, 'et_recall_results_pivot.csv'), index=False)
    g_precision_pivot.to_csv(os.path.join(main_folder, 'golomb_precision_results_pivot.csv'), index=False)
    g_recall_pivot.to_csv(os.path.join(main_folder, 'golomb_recall_results_pivot.csv'), index=False)
    nr_precision_pivot.to_csv(os.path.join(main_folder, 'nr_precision_results_pivot.csv'), index=False)
    nr_recall_pivot.to_csv(os.path.join(main_folder, 'nr_recall_results_pivot.csv'), index=False)
    s_precision_pivot.to_csv(os.path.join(main_folder, 'sudoku_precision_results_pivot.csv'), index=False)
    s_recall_pivot.to_csv(os.path.join(main_folder, 'sudoku_recall_results_pivot.csv'), index=False)

def plot_noise_results(file_path, output_file, metric):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Set the 'Metric' column as the index
    df.set_index('Metric', inplace=True)

    # Transpose the DataFrame to have 'Metric' as columns
    df = df.T

    # Plot the data
    plt.rcParams.update({'font.size': 18})  # You can adjust the font size as needed
    plt.figure(figsize=(6, 8))
    sns.lineplot(data=df, markers=True, dashes=False)

    # Add title and labels
    #plt.title('Performance Metrics Over Different Intervals')
    plt.xlabel('noise percentage')
    plt.ylabel(f'{metric}')

    # Set y-axis range from 0 to 1
    plt.ylim(0, 1)

    # Show legend
    plt.legend(title='Method')

    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, format='png', dpi=300)

    # Show the plot
    plt.show()


from mpl_toolkits.axes_grid1.inset_locator import inset_axes
def new_test_plot_noise():
    noise_types = {'fp_noise', 'fn_noise'}
    metrics = {'precision', 'recall'}

    # Initialize a variable to store the legend handles and labels
    legend_handles = None
    legend_labels = None
    plt.rcParams.update({'font.size': 18})  # You can adjust the font size as needed

    for noise_type in noise_types:
        folder = f'../results/{noise_type}'
        for metric in metrics:
            file_path = f'{folder}/average_{metric}_results_pivot.csv'
            output_file = f'{folder}/{metric}_{noise_type}_col.png'

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Set the 'Metric' column as the index
            df.set_index('Metric', inplace=True)

            # Transpose the DataFrame to have 'Metric' as columns
            df = df.T

            # Plot the data without the legend
            plt.figure(figsize=(6, 8))
            ax = sns.lineplot(data=df, markers=True, dashes=True, linewidth=2.5)

            # Add title and labels
            # plt.title('Performance Metrics Over Different Intervals')
            plt.xlabel('noise percentage')
            plt.ylabel(f'{metric}')

            # Set y-axis range from 0 to 1
            plt.ylim(0, 1)

            # Adjust y-axis ticks
            ax.set_yticks([0, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            ax.set_yticklabels(['0', '0.5', '0.6', '0.7', '0.8', '0.9', '1'])

            # Create inset plot
            #inset_ax = inset_axes(ax, width="40%", height="40%", loc='lower right', borderpad=2)
            #sns.lineplot(data=df, markers=True, dashes=True, linewidth=2.5, ax=inset_ax)
            #inset_ax.set_xlim(0, 4)  # Adjust as needed
            #inset_ax.set_ylim(0.8, 1.0)  # Adjust as needed
            #inset_ax.grid(False)
            #inset_ax.legend().set_visible(False)

            # Extract the legend handles and labels from the first plot
            if legend_handles is None and legend_labels is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

            # Remove legend from the plot
            ax.legend().set_visible(False)
            #ax.grid(True, which='major', linestyle='--', linewidth=0.5)

            plt.tight_layout()

            # Save the plot
            plt.savefig(output_file, format='png', dpi=300)

            # Show the plot
            #plt.show()

    # Adjust the legend labels
    adjusted_labels = {
        'CategoricalNB': 'NB',
        'countcp': 'Count-CP',
        'random_forest': 'RF'
    }
    legend_labels = [adjusted_labels.get(label, label) for label in legend_labels]

    # Create a figure for the legend
    fig_legend, ax_legend = plt.subplots(figsize=(2, 8))

    # Add the legend to the new figure with a box
    legend = fig_legend.legend(legend_handles, legend_labels, loc='center left', ncol=1, frameon=True)

    # Customize the legend box appearance
    legend.get_frame().set_edgecolor('black')  # Set the edge color of the box
    legend.get_frame().set_facecolor('white')  # Set the face color of the box
    legend.get_frame().set_linewidth(1.5)  # Set the line width of the box

    # Remove the axes for the legend plot
    ax_legend.axis('off')

    # Save the legend plot
    fig_legend.savefig('../results/legend_col.png', format='png', dpi=300, bbox_inches='tight')

    # Show the legend plot
    plt.show()


def new_test_plot_noise2():
    noise_types = {'fp_noise', 'fn_noise'}
    #noise_types = {'fp_noise'}

    metrics = {'precision', 'recall'}

    # Initialize a variable to store the legend handles and labels
    legend_handles = None
    legend_labels = None
    sns.set_palette(palette='tab20', n_colors=7)

    plt.rcParams.update({'font.size': 25})  # You can adjust the font size as needed

    for noise_type in noise_types:
        folder = f'../results/{noise_type}'
        for metric in metrics:
            file_path = f'{folder}/average_{metric}_results_pivot.csv'
            output_file = f'{folder}/{metric}_{noise_type}_col_broken.png'

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Set the 'Metric' column as the index
            df.set_index('Metric', inplace=True)

            # Transpose the DataFrame to have 'Metric' as columns
            df = df.T

            # Extract the 'countcp' column
            countcp_col = df.pop('countcp')

            # Concatenate the 'countcp' column back to the DataFrame
            df['countcp'] = countcp_col

            # Create figure and subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6), gridspec_kw={'height_ratios': [5, 1]})

            # Plot the data on both subplots
            sns.lineplot(data=df, markers=True, dashes=True, linewidth=3.5, ax=ax1)
            sns.lineplot(data=df, markers=True, dashes=True, linewidth=3.5, ax=ax2)

            # Set y-axis limits for both subplots
            ax1.set_ylim(0.7, 1)
            ax2.set_ylim(0, 0.1)

            # Hide the spines between ax1 and ax2
            ax1.spines['bottom'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax1.xaxis.tick_top()
            ax1.tick_params(labeltop=False)  # don't put tick labels at the top
            ax2.xaxis.tick_bottom()

            # Add diagonal lines to indicate the break
            d = .015  # how big to make the diagonal lines in axes coordinates
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            ax1.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
            ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

            kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
            ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
            ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

            # Add title and labels
            ax2.set_xlabel('noise percentage')
            ax1.set_ylabel(f'{metric}')
            #ax2.set_ylabel(f'{metric}')

            # Extract the legend handles and labels from the first plot
            if legend_handles is None and legend_labels is None:
                legend_handles, legend_labels = ax1.get_legend_handles_labels()

            # Remove legend from the plots
            ax1.legend().set_visible(False)
            ax2.legend().set_visible(False)

            plt.tight_layout()

            # Save the plot
            plt.savefig(output_file, format='png', dpi=300)

            # Show the plot
            # plt.show()

    # Adjust the legend labels
    adjusted_labels = {
        'CategoricalNB': 'NB',
        'countcp': 'Count-CP',
        'random_forest': 'RF'
    }
    legend_labels = [adjusted_labels.get(label, label) for label in legend_labels]

    # Create a figure for the legend
    fig_legend, ax_legend = plt.subplots(figsize=(2, 8))

    # Add the legend to the new figure with a box
    legend = fig_legend.legend(legend_handles, legend_labels, loc='center left', ncol=1, frameon=True)

    # Customize the legend box appearance
    legend.get_frame().set_edgecolor('black')  # Set the edge color of the box
    legend.get_frame().set_facecolor('white')  # Set the face color of the box
    legend.get_frame().set_linewidth(1.5)  # Set the line width of the box

    # Remove the axes for the legend plot
    ax_legend.axis('off')

    # Save the legend plot
    fig_legend.savefig('../results/legend_col.png', format='png', dpi=300, bbox_inches='tight')

    # Show the legend plot
    #plt.show()


def plot_noise_per_benchmark():
    noise_types = {'fp_noise', 'fn_noise'}
    noise_types = {'fp_noise'}

    metrics = {'precision', 'recall'}
    benchmarks = {'et', 'nr', 'golomb', 'sudoku'}

    sns.set_palette(palette='tab20', n_colors=7)

    # Initialize a variable to store the legend handles and labels
    legend_handles = None
    legend_labels = None

    plt.rcParams.update({'font.size': 25})  # You can adjust the font size as needed

    for noise_type in noise_types:
        folder = f'../results/{noise_type}'
        for benchmark in benchmarks:
            for metric in metrics:
                file_path = f'{folder}/{benchmark}_{metric}_results_pivot.csv'
                output_file = f'{folder}/{benchmark}_{metric}_{noise_type}_col_broken.png'

                # Read the CSV file
                df = pd.read_csv(file_path)

                # Set the 'Metric' column as the index
                df.set_index('Metric', inplace=True)

                # Transpose the DataFrame to have 'Metric' as columns
                df = df.T
                print(df)
                # Extract the 'countcp' column
                countcp_col = df.pop('countcp')
                # Concatenate the 'countcp' column back to the DataFrame
                df['countcp'] = countcp_col
                print(df)
                # Create figure and subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6), gridspec_kw={'height_ratios': [5, 1]})

                # Plot the data on both subplots
                sns.lineplot(data=df, markers=True, dashes=True, linewidth=3.5, ax=ax1)
                sns.lineplot(data=df, markers=True, dashes=True, linewidth=3.5, ax=ax2)

                # Set y-axis limits for both subplots
                ax1.set_ylim(0.5, 1)
                ax2.set_ylim(0, 0.1)

                # Hide the spines between ax1 and ax2
                ax1.spines['bottom'].set_visible(False)
                ax2.spines['top'].set_visible(False)
                ax1.xaxis.tick_top()
                ax1.tick_params(labeltop=False)  # don't put tick labels at the top
                ax2.xaxis.tick_bottom()

                # Add diagonal lines to indicate the break
                d = .015  # how big to make the diagonal lines in axes coordinates
                kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
                ax1.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
                ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

                kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
                ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
                ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

                # Add title and labels
                ax2.set_xlabel('noise percentage')
                ax1.set_ylabel(f'{metric}')
                #ax2.set_ylabel(f'{metric}')

                # Extract the legend handles and labels from the first plot
                if legend_handles is None and legend_labels is None:
                    legend_handles, legend_labels = ax1.get_legend_handles_labels()

                # Remove legend from the plots
                ax1.legend().set_visible(False)
                ax2.legend().set_visible(False)

                plt.tight_layout()

                # Save the plot
                plt.savefig(output_file, format='png', dpi=300)

                # Show the plot
                # plt.show()

    # Adjust the legend labels
    adjusted_labels = {
        'CategoricalNB': 'NB',
        'countcp': 'Count-CP',
        'random_forest': 'RF'
    }
    legend_labels = [adjusted_labels.get(label, label) for label in legend_labels]

    # Create a figure for the legend
    fig_legend, ax_legend = plt.subplots(figsize=(8, 2))

    # Add the legend to the new figure with a box
    legend = fig_legend.legend(legend_handles, legend_labels, loc='center left', ncol=len(legend_labels), frameon=True)

    # Customize the legend box appearance
    legend.get_frame().set_edgecolor('black')  # Set the edge color of the box
    legend.get_frame().set_facecolor('white')  # Set the face color of the box
    legend.get_frame().set_linewidth(1.5)  # Set the line width of the box

    # Remove the axes for the legend plot
    ax_legend.axis('off')
    fig_legend.tight_layout()
    # Save the legend plot
    fig_legend.savefig('../results/legend.png', format='png', dpi=300, bbox_inches='tight')

    # Show the legend plot
    #plt.show()


def plot_results(folder, metric):

    file_path = f'{folder}/{metric}.csv'
    # Read the data from the CSV file
    df = pd.read_csv(file_path)

    # Print the columns to debug
    print("Columns in the DataFrame:", df.columns)

    # Set the 'Benchmark' column as the index
    df.set_index('Classifier', inplace=True)

    # Transpose the DataFrame to have 'Classifier' as columns
    df = df.T

    # Melt the DataFrame for seaborn
    df_melted = df.reset_index().melt(id_vars='index', var_name='Classifier', value_name='Value')
    df_melted.rename(columns={'index': 'Metric'}, inplace=True)

    # Separate the rows with 'Count-CP' classifier
    count_cp_df = df_melted[df_melted['Classifier'] == 'Count-CP']

    # Separate the rows without 'Count-CP' classifier
    other_df = df_melted[df_melted['Classifier'] != 'Count-CP']

    # Concatenate the two DataFrames
    df_melted = pd.concat([other_df, count_cp_df], ignore_index=True)

    print(df_melted)
    # Set the color palette
    sns.set_palette(palette='tab20', n_colors=7)

    # Plot
    plt.rcParams.update({'font.size': 36})  # You can adjust the font size as needed
    plt.figure(figsize=(16, 8))
    bar_plot = sns.barplot(data=df_melted, x='Metric', y='Value', hue='Classifier')

    # Add striped pattern to the special classifier
    for patch, label in zip(bar_plot.patches, df_melted['Classifier']):
        if label == 'Count-CP':
            patch.set_hatch('//')  # More forward slashes = denser pattern
            patch.set_edgecolor('white')  # Make the hatch lines white
            patch.set_linewidth(6)  # Make the hatch lines thicker

    plt.ylabel(f'{metric}')
    plt.xlabel(f'')
    plt.ylim(0.5, 1)
    plt.xticks(rotation=0)

    # Customize the legend position
    handles, labels = bar_plot.get_legend_handles_labels()
    #plt.legend(handles=handles, labels=labels, title='Classifier', loc='upper center', bbox_to_anchor=(0.5, -0.07),
    #           ncol=7)
    plt.legend(handles=handles, labels=labels, loc='upper center',
               bbox_to_anchor=(0.5, -0.07), ncol=7, handletextpad=0.3, columnspacing=0.3, handleheight=0.8, handlelength=1.3)
    #plt.legend(handles=handles, labels=labels, title='Classifier', loc='upper center',
    #           bbox_to_anchor=(0.5, -0.07), ncol=7, handleheight=0.8, handlelength=1.5)

    plt.tight_layout()
    plt.savefig(f'{folder}/{metric}.png', format='png', dpi=500)
    # Show the plot
    plt.show()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--process",action="store_true", default=False)
    parser.add_argument("-m", "--move", action="store_true", default=False)
    parser.add_argument("-pn", "--parse-noise", action="store_true", default=False)
    parser.add_argument("-pln", "--plot-noise", action="store_true", default=False)
    parser.add_argument("-pl", "--plot", action="store_true", default=False)

    args = parser.parse_args()

    if args.process:

        """folder_path = '../results/'
        benchmark = 'sudoku'
        classifier = 'CN2'
        #output_csv = 'results.csv'
    
        process_files(folder_path, benchmark, classifier)"""

        all_benchmarks = ["sudoku", "golomb", "exam_timetabling", "nurse_rostering"]
        all_classifiers = ["random_forest", "MLP", "CategoricalNB", "DT", "KNN", "CN2", "countcp"]

        folder_path = '../results/no_noise/'
        for benchmark in all_benchmarks:
            for classifier in all_classifiers:
                process_files(folder_path, benchmark, classifier)

    elif args.move:

        # Get the current directory
        current_directory = "../results/"
        # Specify the target directory
        target_directory = "../results/no_noise/"
        # Specify the substring to search for
        substring_to_search = "random"

        move_files_with_substring(current_directory, target_directory, substring_to_search)

    elif args.parse_noise:
        # Define the main folder path
        folder = '../results/fp_noise/'
        parse_noise_results(folder)
    elif args.plot_noise:
        #new_test_plot_noise2()
        plot_noise_per_benchmark()

    else:
        folder_path = '../results/no_noise/'
        metric = 'Recall'
        plot_results(folder_path, metric)


