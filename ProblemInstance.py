import numpy as np
from cpmpy.expressions.utils import all_pairs

from utils.generic_utils import get_variables_from_constraints, get_var_name, get_var_dims, get_divisors, vars_in_rel, \
    generate_expression, get_combinations, combine_sets_distinct, get_scope, get_var_ndims, average_difference, \
    get_symbolic_parameter, get_constant, get_var_partition_idx


class ProblemInstance:

    def __init__(self, constraints=None, variables=None, params=None, name=None, custom_partitions=None):

        if constraints is None:
            constraints = []
        self.constraints = constraints
        if variables is None:
            variables = []
        self.variables = variables
        if params is None:
            params = dict()
        self.params = params
        if name is None:
            name = ""
        self.name = name

        if custom_partitions is None:
            custom_partitions = []
        self.custom_partitions = custom_partitions

        assert isinstance(constraints,list), "'constraints' argument in ProblemInstance should be a list of constraints"
        assert isinstance(variables,list), "'variables' argument in ProblemInstance should be a list of variables"
        assert isinstance(params, dict), "'params' argument in ProblemInstance should be a dictionary of parameters"

        if len(self.constraints) > 0 and len(self.variables) == 0:
            self.init_vars()

        self._var_names = []
        self._dim_lengths = []
        self._dim_divisors = []
        self._divisors_indices = []
        self._lang = []
        self.init_lang()

        self._datasetX = []
        self._datasetY = []

    def init_vars(self):
        self.variables = get_variables_from_constraints(self.constraints)

    def init_dimensions_features(self):

        #### Needed for the features ####
        ## for input vars
        # Length of dimensions per variable name
        self._var_names = list(set([get_var_name(x) for x in self.variables]))
        var_dims = [[get_var_dims(x) for x in self.variables if get_var_name(x) == self._var_names[i]] for i in
                    range(len(self._var_names))]
        self._dim_lengths = [
            [np.max([var_dims[i][k][j] for k in range(len(var_dims[i]))]) + 1 for j in range(len(var_dims[i][0]))] for i
            in range(len(var_dims))]

        self._dim_divisors = []
        self._divisors_indices = []

        for i in range(len(self._dim_lengths)):
            dim_divisors = []
            self._divisors_indices.append([])
            for j in range(len(self._dim_lengths[i])):
                self._divisors_indices[i].append([])
                # TODO: it works but not implemented correctly... Need to do it once for max, not for all i ... It now works because we have one type of variable

                divisors = get_divisors(self._dim_lengths[i][j])
                divisors_used = []
                block = 0
                for di in range(len(divisors)):
                    # find all keys for matching values
                    matching_keys = [key for key, value in self.params.items() if value == divisors[di]]
                    if len(matching_keys) == 0:
                        continue
                    # extend self._divisors_indices[i][j] with all matching keys
                    self._divisors_indices[i][j].extend(matching_keys)
                    for _ in range(len(matching_keys)):
                        block += 1
                        divisors_used.append(divisors[di])

                dim_divisors.append(divisors_used)

            self._dim_divisors.append(dim_divisors)

    def init_lang(self):
        self._lang = list(set([c.get_relation() for c in self.constraints]))  # Language: relations derived from constraints

    def set_lang(self, lang):
        self._lang = lang

    def set_constraints(self, constraints):
        self.constraints = constraints

    def set_variables(self, variables):
        self.variables = variables

    def set_params(self, params):
        self.params = params

    def set_dataset(self, datasetX, datasetY):
        self._datasetX = datasetX
        self._datasetY = datasetY

    def set_custom_partitions(self, custom_partitions):
        self.custom_partitions = custom_partitions

    def get_constraints(self):
        return self.constraints

    def get_variables(self):
        return self.variables

    def get_params(self):
        return self.params

    def get_lang(self):
        if len(self._lang) == 0:
            self.init_lang()
        return self._lang

    def get_dim_lengths(self):
        return self._dim_lengths

    def get_divisors(self):
        return self._dim_divisors

    def get_divisor_indices(self):
        return self._divisors_indices

    def get_dataset(self):
        if len(self._datasetX) == 0:
            self.create_instance_dataset()
        return self._datasetX, self._datasetY

    def get_custom_partitions(self):
        return self.custom_partitions

    def transform_params(self):
        self.params.update({f'{key}-1': v - 1 for key,v in self.params.items()})
        self.params.update({"0": 0, "1": 1})
        self.params.update({f'-({key})': -v for key, v in self.params.items() if key != '0'})

    def update_divisors_from_instance(self, other_instance):
        assert isinstance(other_instance, ProblemInstance)
        # _dim_divisors and _divisors_indices already initialized.

        updated_dim_divisors = []
        updated_divisors_indices = []

        other_divisor_indices = other_instance.get_divisor_indices()

        for i in range(len(other_divisor_indices)):

            new_dim_divisors = []
            new_divisors_indices = []

            for j in range(len(other_divisor_indices[i])):
                new_divisors = []
                new_indices = []

                for divisor_ind in other_divisor_indices[i][j]:
                    # Check if the parameters for the input instance divisor exist in this instance
                    matched = divisor_ind in self._divisors_indices[i][j]

                    if matched:
                        # If there are matching parameters, keep this divisor and its parameters
                        new_divisors.append(self.params[divisor_ind])
                        new_indices.append(divisor_ind)
                    else:
                        new_divisors.append(1)
                        new_indices.append('1')

                new_dim_divisors.append(new_divisors)
                new_divisors_indices.append(new_indices)

            updated_dim_divisors.append(new_dim_divisors)
            updated_divisors_indices.append(new_divisors_indices)

        # Update the instance's attributes with the filtered lists
        self._dim_divisors = updated_dim_divisors
        self._divisors_indices = updated_divisors_indices

    def update_target_divisors_from_instance(self, other_instance):
        assert isinstance(other_instance, ProblemInstance)
        # _dim_divisors and _divisors_indices already initialized.

        updated_dim_divisors = []
        updated_divisors_indices = []

        other_divisor_indices = other_instance.get_divisor_indices()

        for i in range(len(other_divisor_indices)):

            new_dim_divisors = []
            new_divisors_indices = []

            for j in range(len(other_divisor_indices[i])):
                new_divisors = []
                new_indices = []

                for divisor_ind in other_divisor_indices[i][j]:
                    # Check if the parameters for the input instance divisor exist in this instance
                    matched = divisor_ind in self._divisors_indices[i][j]

                    if matched:
                        # If there are matching parameters, keep this divisor and its parameters
                        new_divisors.append(self.params[divisor_ind])
                        new_indices.append(divisor_ind)
                    else:
                        new_divisors.append(1)
                        new_indices.append('1')

                new_dim_divisors.append(new_divisors)
                new_divisors_indices.append(new_indices)

            updated_dim_divisors.append(new_dim_divisors)
            updated_divisors_indices.append(new_divisors_indices)

        # Update the instance's attributes with the filtered lists
        self._dim_divisors = updated_dim_divisors
        self._divisors_indices = updated_divisors_indices

    def gen_bias(self):

        for r in self._lang:
            arity = vars_in_rel(r, 'var')
            constants = vars_in_rel(r, 'const')
            if constants > 0:
                consts = [[param] * constants for param in self.params.values()]
                consts = [list(x) for x in dict.fromkeys([tuple(sublist) for sublist in consts])]
            else:
                consts = [()]

            for const in consts:
                if arity == 2:
                    for v1, v2 in all_pairs(self.variables):
                        c, _ = generate_expression(r, [v1, v2], list(const))
                        yield c
                elif arity == 4:
                    combinations = get_combinations(self.variables, 2)
                    result_combinations = combine_sets_distinct(combinations, combinations)
                    for ((v1, v2), (v3, v4)) in result_combinations:
                        c, _ = generate_expression(r, [v1, v2, v3, v4])
                        yield c

    def get_con_features(self, c):

        # get instance parameters etc
        params = self.params
        var_names = self._var_names
        dim_divisors = self._dim_divisors

        # Returns the features associated with constraint c
        feature_representations = [[]]

        scope = get_scope(c)
        var_name = get_var_name(scope[0])
        var_name_same = all([(get_var_name(var) == var_name) for var in scope])
        feature_representations[0].append(str(var_name_same))

        # Global var dimension properties
        vars_ndims = [get_var_ndims(var) for var in scope]
        ndims_max = max(vars_ndims)

        vars_dims = [get_var_dims(var) for var in scope]
        dim = []
        for j in range(ndims_max):
            dim.append([vars_dims[i][j] for i in range(len(vars_dims)) if len(vars_dims[i]) > j])
            dimj_has = len(dim[j]) > 0

            if dimj_has:
                dimj_same = all([dim_temp == dim[j][0] for dim_temp in dim[j]])
                [features.append(str(dimj_same)) for features in feature_representations]
                dimj_diff = average_difference(dim[j])
                dimj_diff_symbolic = get_symbolic_parameter(dimj_diff, params)
                # Iterate over each value returned by get_symbolic_parameter()
                new_feature_repr = []
                [new_feature_repr.append(feature_list.copy() + [dimj_diff]) for feature_list in feature_representations
                 for dimj_diff in dimj_diff_symbolic]
                feature_representations = new_feature_repr


            else:
                [features.append("True") for features in feature_representations]
                # TODO: Not sure what I am doing here
                [features.append(len(params)+3) for features in feature_representations]

            # divisors features --- per var_name, per dimension (up to max dimensions), per divisor (up to max divisors)
            # block same, max, min, average, spread
            # for each variable name (type of variable)
            for var_name in range(len(var_names)):

                vars_block = [[dim[j][var] // divisor for var in range(len(scope))
                               if var_names[var_name] == get_var_name(scope[var])]
                              for divisor in dim_divisors[var_name][j]]

                for l in range(len(dim_divisors[var_name][j])):

                    block_same = all([vars_block[l][var] == vars_block[l][0] for var in range(len(vars_block[l]))])
                    [features.append(str(block_same)) for features in feature_representations]
                    block_diff_sumbolic = get_symbolic_parameter(average_difference(vars_block[l]), params)
                    new_feature_repr = []
                    [new_feature_repr.append(feature_list.copy() + [block_diff]) for feature_list in
                     feature_representations for block_diff in block_diff_sumbolic]
                    feature_representations = new_feature_repr


        vars_partitions = [get_var_partition_idx(var, self.custom_partitions) for var in scope]
        par = []
        for j in range(len(self.custom_partitions)):
            par.append([vars_partitions[i][j] for i in range(len(vars_partitions)) if len(vars_partitions[i]) > j])
            parj_has = len(par[j]) > 0

            if parj_has:
                parj_same = all([par_temp == par[j][0] for par_temp in par[j]])
                [features.append(str(parj_same)) for features in feature_representations]
                parj_diff = average_difference(par[j])
                parj_diff_symbolic = get_symbolic_parameter(parj_diff, params)
                # Iterate over each value returned by get_symbolic_parameter()
                new_feature_repr = []
                [new_feature_repr.append(feature_list.copy() + [parj_diff]) for feature_list in feature_representations
                 for parj_diff in parj_diff_symbolic]
                feature_representations = new_feature_repr

        con_in_gamma = -1
        con_relation = c.get_relation()
        for i in range(len(self._lang)):
            if self._lang[i] == con_relation:
                con_in_gamma = i
                break

        if con_in_gamma == -1:
            raise Exception("Check why constraint relation was not found in relations")
        [features.append(con_in_gamma) for features in feature_representations]

        num = get_constant(c)
        has_const = len(num) > 0
        [features.append(str(has_const)) for features in feature_representations]

        if has_const:
            consts = get_symbolic_parameter(int(num[0]), params)
            # take the index of the constant in the parameters
            new_feature_repr = []
            [new_feature_repr.append(feature_list.copy() + [const]) for feature_list in
             feature_representations for const in consts]
            feature_representations = new_feature_repr
            #[features.append() for features in feature_representations]
        else:
            [features.append("NaN") for features in feature_representations]

        return feature_representations


    def create_instance_dataset(self):

        setModel = set(self.constraints)

        for c in self.gen_bias():
            features = self.get_con_features(c)
            self._datasetX.extend(features)
            if c in setModel:
                self._datasetY.extend([1]*len(features))
            else:
                self._datasetY.extend([0]*len(features))

    def noisify_dataset(self, p, mode):
        assert 0 < p < 1, "p must be a percentage"
        assert mode in ["fp", "fn", "both"], "Define a valid mode: 'fp' for false positives, " \
                                                "'fn' for false negatives" \
                                                "'both' for both"
        assert len(self._datasetY) > 0, "Dataset must be initialized in order to noisify"
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
            # Calculate the number of negative indices to change
            num_neg_noisy = int(p * len(np.where(labels == 1)[0]))
            if len(neg_indices) > 0:
                # Randomly choose neg_indices to change
                neg_noisy_indices = np.random.choice(neg_indices, size=num_neg_noisy, replace=False)
                # Change 0 to 1 for the chosen neg_indices
                noisy_labels[neg_noisy_indices] = 1

        self._datasetY = noisy_labels

    def noisify_constraints(self, p, mode):
        assert 0 < p < 1, "p must be a percentage"
        assert mode in ["fp", "fn", "both"], "Define a valid mode: 'fp' for false positives, " \
                                                "'fn' for false negatives" \
                                                "'both' for both"

        labels = []
        B = []
        # Create all candidates
        setModel = set(self.constraints)
        for c in self.gen_bias():
            B.append(c)
            if c in setModel:
                labels.append(1)
            else:
                labels.append(0)

        labels = np.array(labels)
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
            # Calculate the number of negative indices to change
            num_neg_noisy = int(p * len(np.where(labels == 1)[0]))
            if len(neg_indices) > 0:
                # Randomly choose neg_indices to change
                neg_noisy_indices = np.random.choice(neg_indices, size=num_neg_noisy, replace=False)
                # Change 0 to 1 for the chosen neg_indices
                noisy_labels[neg_noisy_indices] = 1

        pos_indices = np.where(noisy_labels == 1)[0]
        B = np.array(B)
        self.constraints = B[pos_indices]

    def __str__(self):
        return f"ProblemInstance {self.name}, with parameters {self.params}.\nvariables: {self.variables}\nConstraints: {self.constraints}"
