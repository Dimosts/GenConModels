import argparse
import copy
import gc
import time

from cpmpy.expressions.utils import all_pairs
from sklearn.model_selection import LeaveOneOut, LeavePOut

from GenAcq import GenAcq
from GeneralizedCon.GenModel import GenModel
from Metrics import Metrics
from ProblemInstance import ProblemInstance
from count_cp_simulate import CountCP

from sklearn import tree

from utils.experiment_utils import get_benchmark_models, construct_classifier, save_results, load_instances
from utils.generic_utils import *

all_benchmarks = ["sudoku", "golomb", "exam_timetabling", "nurse_rostering"]
all_classifiers = ["random_forest", "MLP", "GaussianNB", "CategoricalNB", "SVM", "DT", "KNN", "CN2", "countcp"]

def parse_args():
    parser = argparse.ArgumentParser(description='Run AAAI experiments with various classifiers and benchmarks.')

    # Experiment Configuration Group
    exp_group = parser.add_argument_group('Experiment Configuration')
    exp_group.add_argument("-b", "--benchmark", type=str, 
                          choices=all_benchmarks,
                          help="Benchmark to use")
    exp_group.add_argument("-ts", "--training-size", type=int, default=1,
                          help="Number of training instances for reverse leave-p-out cross validation")
    exp_group.add_argument("-cp", "--custom-partitions", action="store_true", default=False,
                          help="Use custom partitions")

    # Classifier Configuration Group
    clf_group = parser.add_argument_group('Classifier Configuration')
    clf_group.add_argument("-c", "--classifier", type=str, 
                          choices=all_classifiers,
                          help="Machine learning classifier to use")

    # Noise Configuration Group
    noise_group = parser.add_argument_group('Noise Configuration')
    noise_group.add_argument("-n", "--noise_mode", type=str, 
                            choices=["fp", "fn"],
                            help="Noise mode to use (fp: false positive, fn: false negative)")
    noise_group.add_argument("-p", "--percentage", type=float, 
                            choices=[0, 0.05, 0.1, 0.15, 0.2],
                            help="Noise percentage to apply")
    
    return parser.parse_args()


def aaai_experiments(classifiers=None, benchmarks=None, noise_mode=None, percentage=None, training_size=None,
                     custom_par=False):
    if classifiers is None:
        classifiers = all_classifiers
    else:
        classifiers = [classifiers]
    if benchmarks is None:
        benchmarks = all_benchmarks
    else:
        benchmarks = [benchmarks]
    if percentage is None:
        percentage = [0, 0.05, 0.1, 0.15, 0.2]
    else:
        percentage = [percentage]
    if noise_mode is None:
        noise_mode = ["fp", "fn"]
        percentage = [0, 0.05, 0.1, 0.15, 0.2]
    else:
        noise_mode = [noise_mode]
    if training_size is None:
        training_size = 1

    for bench in benchmarks:
        print(f"## Experiments for Benchmark {bench} ----------")
        instances = load_instances(bench)
        for classifier_name in classifiers:
            classifier = construct_classifier(classifier_name)
            print(f"\n\n### Classifier {classifier_name} --------")

            loo = LeavePOut(training_size)
            instances = np.array(instances)

            for noise in noise_mode:
                print(f"\n#### Noise Mode {noise} --------")
                for p in percentage:  # noise percentage
                    print(f"\n##### Noise Percentage {p} ------")
                    for test, train in loo.split(instances):
                        train_instances = instances[train]
                        inputInstances = []
                        print(f"\n###### Leave p out - Training instances {train} ----")
                        for train_instance in train_instances:
                            train_CSP = train_instance['csp']
                            train_params = train_instance['params']
                            if custom_par:
                                train_partitions = [train_instance['partitions']]
                            else:
                                train_partitions = []
                            inputInstance = ProblemInstance(constraints=train_CSP.copy(), params=train_params.copy(),
                                                            name=f"{bench}_input",custom_partitions=train_partitions)
                            inputInstances.append(inputInstance)

                        ### Generalize based on input model
                        if classifier_name == "countcp":
                            ga = CountCP(inputInstances=inputInstances)
                        else:
                            ga = GenAcq(inputInstances=inputInstances, classifier=classifier,
                                    classifier_name=classifier_name)

                        if p > 0:
                            ga.noisify(p, noise)

                        # train classifier
                        start = time.time()

                        print(f"----- Training Classifier -----")
                        ga.learn()
                        if classifier_name == "DT":
                            text_representation = tree.export_text(ga.get_classifier(),
                                                                   feature_names=ga.get_encoded_feature_names())
                            print(text_representation)

                            rules = get_rules_from_dt(classifier, ga.get_encoded_feature_names(),
                                                      ga.get_classifier().classes_)
                            print("rules: ")
                            for r in rules:
                                print(r)
                            pos_rules = []

                            print("positive rules: ")
                            for r in rules:
                                if 'then 1' in r:
                                    print(r)
                                    pos_rules.append(r)

                            feature_mapping = ga.get_feature_mapping()
                            feature_categories = ga.get_feature_categories()
                            feature_names = ga.get_feature_names()
                            encoders = ga._label_encoders
                            gen_model = GenModel(lang=inputInstances[0].get_lang(),
                                                 params=inputInstances[0].get_params(),
                                                 rules=pos_rules, feature_mapping=feature_mapping,
                                                 feature_categories=feature_categories)
                            gen_model.create_gen_model_from_dt(feature_names=feature_names, encoders=encoders)
                            print("Constraint Specifications: ")
                            for c in gen_model.gen_constraints:
                                print(c)
                        elif classifier_name == "CN2":
                            rules = ga.get_classifier().rule_list
                            feature_mapping = ga.get_feature_mapping()
                            feature_categories = ga.get_feature_categories()

                            new_rules = []
                            prec_neg_rules = []
                            for rule in rules:
                                if "class=1" in str(rule):
                                    class_distr = rule.curr_class_dist.tolist()
                                    if class_distr[0] < class_distr[1]:
                                        #print(rule, rule.curr_class_dist.tolist())
                                        new_rules.append(rule)

                            gen_model = GenModel(lang=inputInstances[0].get_lang(),
                                                 params=inputInstances[0].get_params(),
                                                 rules=new_rules, feature_mapping=feature_mapping,
                                                 feature_categories=feature_categories)
                            gen_model.create_gen_model_from_cn2()

                        end = time.time()
                        training_time = end - start
                        print(f"Training Time: {training_time}")
                        for t in test:
                            print(f"\n ####### Testing on Instance {t} -----")

                            metrics = Metrics()

                            test_instance = instances[t]
                            test_CSP = test_instance['csp']
                            test_params = test_instance['params']
                            if custom_par:
                                test_partitions = [test_instance['partitions']]
                            else:
                                test_partitions = []

                            print(f"test_params: {test_params}, {test_instance['name']}")

                            targetInstance = ProblemInstance(constraints=test_CSP.copy(), params=test_params.copy(),
                                                             name=f"{bench}_target{t}", custom_partitions=test_partitions)
                            ga.set_target_instance(targetInstance=targetInstance)

                            print(f"----- Generating ground constraints -----")
                            start = time.time()
                            # Generate constraints for target instance
                            # --- Extract generalized constraints for DT and CN2 ---
                            # --- Generate and test for the rest classifiers ---
                            if classifier_name == "DT":

                                C_L = gen_model.generate_ground_model(targetInstance)
                                C_L = list(dict.fromkeys(C_L))
                            elif classifier_name == "CN2":

                                C_L = gen_model.generate_ground_model(targetInstance)
                                C_L = list(dict.fromkeys(C_L))

                                metrics._C_L = C_L
                                metrics._C_T = test_CSP

                                # Evaluate method

                                if len(C_L) < 1000:
                                    metrics.evaluate()
                                    C_L = ga.generalize()
                            else:
                                C_L = ga.generalize()

                            end = time.time()
                            prediction_time = end - start
                            print(f"----- Generation time: {prediction_time} -----")
                            metrics._C_L = C_L
                            metrics._C_T = test_CSP

                            # Evaluate method
                            metrics.evaluate()
                            metrics.set_times(training_time, prediction_time)

                            if p == 0:
                                save_results(clf=classifier_name, bench=bench, metrics=metrics,
                                             instance=[len(train_CSP), len(test_CSP)])
                            else:
                                save_results(clf=classifier_name, bench=bench, metrics=metrics,
                                             instance=[len(train_CSP), len(test_CSP)], noise_p=p, noise_mode=noise)


if __name__ == "__main__":
    args = parse_args()

    aaai_experiments(args.classifier, args.benchmark, args.noise_mode, args.percentage, args.training_size,
                     args.custom_partitions)
