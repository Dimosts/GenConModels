import Orange
from Orange import classification
from cpmpy import *

from ProblemInstance import ProblemInstance
from utils.generic_utils import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class GenAcq:

    # _target_instance: ProblemInstance
    # _input_instances: list[ProblemInstance]

    def __init__(self, inputInstances=None, classifier=None, classifier_name=None, targetInstance=None):

        self._target_instance = None
        self._input_instances = None
        self._lang = None

        self._classifier = classifier  # the classifier to be used
        self._classifier_name = classifier_name  # the name of the classifier to be used

        self.set_input_instances(inputInstances)
        self.set_target_instance(targetInstance)

        self._datasetX = []
        self._datasetY = []

    def _init_features(self):

        dim_lengths = self._input_instances[0].get_dim_lengths()
        dim_divisors = self._input_instances[0].get_divisor_indices()
        params = self._input_instances[0].get_params()

        self._feature_names = ["var_name_same"]
        self._feature_categories = [['False', 'True']]
        self._feature_mapping = [{"Partition": "name"}]

        for i in range(len(dim_lengths)):
            for j in range(len(dim_lengths[i])):
                self._feature_names.append(f"Dim{j}_same")
                self._feature_categories.append(['False', 'True'])
                self._feature_mapping.append({"Partition": "dim", "Condition": "same", "dim": j})

                self._feature_names.append(f"Dim{j}_avg_diff")
                self._feature_categories.append(list(params.keys()) + ["NaN"])
                self._feature_mapping.append({"Sequence": "dim", "Condition": "diff", "dim": j})

                block = 0
                for di in dim_divisors[i][j]:
                    block += 1
                    self._feature_names.append(f"Dim{j}_block{block}_same")
                    self._feature_categories.append(['False', 'True'])
                    self._feature_mapping.append({"Partition": "block", "Condition": "same", "dim": j, "divisor": di})

                    self._feature_names.append(f"Dim{j}_block{block}_avg_diff")
                    self._feature_categories.append(list(params.keys()) + ["NaN"])
                    self._feature_mapping.append({"Sequence": "block", "Condition": "diff", "dim": j, "divisor": di})

                   
        for j in range(len(self.get_input_instances()[0].get_custom_partitions())):
            self._feature_names.append(f"custom_par{j}_same")
            self._feature_categories.append(['False', 'True'])
            self._feature_mapping.append({"Partition": "custom", "Condition": "same", "par": j})

            self._feature_names.append(f"custom_par{j}_avg_diff")
            self._feature_categories.append(list(params.keys()) + ["NaN"])
            self._feature_mapping.append({"Sequence": "custom_par", "Condition": "diff", "par": j})

        self._feature_names.append(f"Relation")
        self._feature_categories.append(list(range(len(self._lang))))
        self._feature_mapping.append({"Relation": "lang"})

        self._feature_names.append(f"Has_Constant")
        self._feature_categories.append(['False', 'True'])
        self._feature_mapping.append({"Relation": "has_constant"})

        self._feature_names.append(f"Constant_parameter")
        self._feature_categories.append(list(params.keys()) + ["NaN"])
        self._feature_mapping.append({"Relation": "constant_param"})

        # Initialize the encoder
        self._label_encoders = []
        self._encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self._encoded_feature_names = self._feature_names

    def aggregate_divisors(self):
        """
        First, update the first ProblemInstance object with information from all others.
        Then, update all other objects based on the first one.
        """
        instances = self._input_instances

        if not instances:
            raise Exception("Tried to aggregate divisors, without having input instances")  # Handle empty list case

        # Step 1: Aggregate updates into the first instance
        for i in range(1, len(instances)):
            instances[0].update_divisors_from_instance(instances[i])

        # Step 2: Distribute updates from the first instance to all others
        for i in range(1, len(instances)):
            instances[i].update_divisors_from_instance(instances[0])

    def set_input_instances(self, inputInstances):
        self._input_instances = inputInstances

        if self._input_instances is not None:
            if not isinstance(self._input_instances, list):
                self._input_instances = [self._input_instances]

            self._lang = self._input_instances[0].get_lang()  # Language: relations derived from input model

            # transform symbolic parameters for the variables
            for i in range(len(self._input_instances)):
                self._input_instances[i].init_dimensions_features()
                self._input_instances[i].transform_params()

            self.aggregate_divisors()
            self._init_features()

    def get_input_instances(self):
        return self._input_instances

    def set_target_instance(self, targetInstance):
        self._target_instance = targetInstance

        if self._target_instance is not None:
            self._target_instance.transform_params()
            self._target_instance.init_dimensions_features()
            self._target_instance.update_target_divisors_from_instance(self._input_instances[0])

    def get_target_instance(self):
        return self._target_instance

    def set_classifier(self, classifier):
        self._classifier = classifier

    def get_classifier(self):
        return self._classifier

    def set_feature_names(self, feature_names):
        self._feature_names = feature_names

    def get_feature_names(self):
        return self._feature_names

    def get_encoded_feature_names(self):
        return self._encoded_feature_names

    def get_feature_mapping(self):
        return self._feature_mapping

    def get_feature_categories(self):
        return self._feature_categories

    def set_dataset(self, datasetX, datasetY):
        self._datasetX = datasetX
        self._datasetY = datasetY

    def get_dataset(self):
        return self._datasetX, self._datasetY

    def create_input_dataset(self):

        for instance in self._input_instances:
            datasetX, datasetY = instance.get_dataset()
            self._datasetX.extend(datasetX)
            self._datasetY.extend(datasetY)

        # Fit and transform the data with label encoding
        for i in range(len(self._datasetX[0])):
            le = LabelEncoder()
            le.fit(self._feature_categories[i])
            self._label_encoders.append(le)

    def get_target_dataset(self):

        datasetX, datasetY = self._target_instance.get_dataset()

        # Transform the data with the label encoder
        datasetX = np.array(datasetX)
        for i in range(len(datasetX[0])):
            le = self._label_encoders[i]
            datasetX[:, i] = le.transform(datasetX[:, i])

        datasetX = np.array(datasetX, dtype=np.int64)
        return datasetX, datasetY

    def noisify(self, p, mode):
        assert 0 < p < 1, "p must be a percentage"
        assert mode in ["fp", "fn", "both"], "Define a valid mode: 'fp' for false positives, " \
                                             "'fn' for false negatives" \
                                             "'both' for both"
        if len(self._datasetY) == 0:
            self.create_input_dataset()

        labels = np.array(self._datasetY)
        noisy_labels = labels.copy()
        if mode in ('fn', 'both'):
            # Get indices for positive labels
            pos_indices = np.where(labels == 1)[0]
            # Calculate the number of positive indices to change
            num_pos_noisy = int(p * len(pos_indices))
            # Randomly choose pos_indices to change
            pos_noisy_indices = np.random.choice(pos_indices, size=num_pos_noisy, replace=False)
            # Change 1 to 0 for the chosen pos_indices
            noisy_labels[pos_noisy_indices] = 0

        if mode in ('fp', 'both'):
            # Get indices for negative labels
            neg_indices = np.where(labels == 0)[0]
            # Calculate the number of negative indices to change based on the percentage of the true set
            num_neg_noisy = int(p * len(np.where(labels == 1)[0]))
            print("num_neg_noisy: ", num_neg_noisy)
            if len(neg_indices) > 0:
                # Randomly choose neg_indices to change
                neg_noisy_indices = np.random.choice(neg_indices, size=num_neg_noisy, replace=False)
                # Change 0 to 1 for the chosen neg_indices
                noisy_labels[neg_noisy_indices] = 1

        self._datasetY = noisy_labels

    def learn(self):
        if len(self._datasetY) == 0:
            print(f"--- Creating input dataset ---")
            self.create_input_dataset()
        # assert len(self._datasetY) > 0, "Dataset must be initialized in order to train classifier"

        # Train the specified ML classifier
        if self._classifier_name == "GaussianNB":  # If GNB, try to use class weights, to handle the imbalance
            # (doesn't really improve though)
            self._classifier.fit(self._datasetX, self._datasetY, sample_weight=compute_sample_weights(self._datasetY))
        elif self._classifier_name == "CategoricalNB":  # If CNB, define the number of categories
            self._classifier.set_params(**{"min_categories": [len(f) for f in self._feature_categories]})
            datasetX = np.array(self._datasetX)
            new_datasetX = np.ndarray(shape=datasetX.shape)
            for i in range(len(self._datasetX[0])):
                new_datasetX[:, i] = self._label_encoders[i].transform(datasetX[:, i])
            self._datasetX = new_datasetX
            self._classifier.fit(self._datasetX, self._datasetY)
        elif self._classifier_name == "CN2":
            # Convert all elements in X to strings
            X_str = [[str(value) for value in instance] for instance in self._datasetX]
            Y_str = [str(label) for label in self._datasetY]

            # Create the list of DiscreteVariable objects for the features
            categories = [[str(c) for c in cat] for cat in self._feature_categories]
            feature_vars = [Orange.data.DiscreteVariable(f'{self._feature_names[i]}', values=categories[i]) for i, vals
                            in
                            enumerate(self._feature_names)]

            # Determine the unique classes for the class variable
            unique_classes = sorted(set(Y_str))
            class_var = Orange.data.DiscreteVariable('class', values=unique_classes)

            # Create a domain with the feature variables and the class variable
            self.domain = Orange.data.Domain(feature_vars, class_var)
            XY = [X_str[i] + [Y_str[i]] for i in range(len(Y_str))]

            # Construct the Table using the domain and the data
            data = Orange.data.Table.from_list(self.domain, XY)

            # Initialize CN2 learner and create the classifier
            learner = Orange.classification.CN2UnorderedLearner()
            # learner.rule_finder.alpha = 0.2  # Example value, adjust based on dataset
            learner.rule_finder.min_covered_examples = 50  # Adjust as needed

            self._classifier = learner(data)

        else:
            datasetX = np.array(self._datasetX)
            new_datasetX = np.ndarray(shape=datasetX.shape)
            for i in range(len(self._datasetX[0])):
                new_datasetX[:, i] = self._label_encoders[i].transform(datasetX[:, i])
            self._datasetX = new_datasetX
            self._classifier.fit(self._datasetX, self._datasetY)

    def generalize(self):

        C_L = []  # learned set of constraints
        for c in self._target_instance.gen_bias():  # generate bias by generating constraints one by one
            features = self._target_instance.get_con_features(c)
            if self._classifier_name not in ["CN2"]:
                # get encoded features of constraints for target problem
                features = np.array(features)
                new_features = np.ndarray(shape=features.shape)
                for i in range(len(features[0])):
                    le = self._label_encoders[i]
                    try:
                        new_features[:, i] = le.transform(features[:, i])
                    except ValueError:
                        raise Exception(
                            f"Issue in label encoding! encoding features in dimension {i}: {features[:, i]} \nCategories: {self._feature_categories[i]}")
                # predict if it is part of the target problem
                y = self._classifier.predict(new_features)
            else:
                new_X_str = [[str(value) for value in feature] for feature in features]
                # new values and undefined class
                new_instances = [Orange.data.Instance(self.domain, X + [None]) for X in new_X_str]
                y = [self._classifier(instance) for instance in new_instances]
            if sum(y) >= 1:
                C_L.append(c)
        #                if c not in self._target_model:
        #                    print(f"constraint {c} not in C_T, features: {features}")
        #                    input()
        print("generalized -----")
        return C_L
