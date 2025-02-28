from utils.generic_utils import get_var_dims, average_difference, get_symbolic_parameter, get_var_partition_idx, \
    get_var_name


class Condition:

    def __init__(self, name, feature_map, condition=None):
        assert isinstance(name, str), "name given to Condition should be a string"
        assert isinstance(feature_map, dict), "feature_map given to Condition should be a dictionary"
        self.name = name
        self.condition = condition

        self.dim = None
        if "dim" in feature_map.keys():
            self.dim = feature_map["dim"]

        self.par = None
        if "par" in feature_map.keys():
            self.par = feature_map["par"]

        self.divisor = None
        if "divisor" in feature_map.keys():
            self.divisor = feature_map["divisor"]

    def check_condition(self, comb, params=None):
        raise NotImplementedError("Generic check_condition function was called")

    def __str__(self):
        return f"<Condition {self.name} - Dimension {self.dim}>"

    def __repr__(self):
        return self.__str__()

class NotSameCondition(Condition):
    pass


class DiffCondition(Condition):

    def __init__(self, name, feature_map, categ, condition=None):
        super().__init__(name, feature_map, condition)
        self.allowed_diff = categ

    def __str__(self):
        return f"<Condition {self.name} - Dimension {self.dim} - Allowed diff: {self.allowed_diff} - Original Condition: {self.condition}>"


class NotSameDim(NotSameCondition):

    def __init__(self, feature_map):
        super().__init__(f"not_same_dim_{feature_map['dim']}", feature_map)

    def check_condition(self, comb, params=None, custom_partitions=None):
        dims_list = [get_var_dims(var) for var in comb]
        specific_dim_indices = [dims[self.dim] for dims in dims_list]
        # Check if all indices in the given dimension are unique
        return len(specific_dim_indices) == len(set(specific_dim_indices))


class NotSameBlock(NotSameCondition):

    def __init__(self, feature_map):
        assert "divisor" in feature_map.keys(), "NotSameBlock condition attempted to be created without a divisor parameter"
        super().__init__(f"not_same_dim_{feature_map['dim']}_block_{feature_map['divisor']}", feature_map)

    def check_condition(self, comb, params=None, custom_partitions=None):
        assert isinstance(params, dict)
        dims_list = [get_var_dims(var) for var in comb]
        specific_dim_indices = [dims[self.dim] for dims in dims_list]
        block_size = params[self.divisor]
        dim_block_indices = [dim // block_size for dim in specific_dim_indices]

        # Check if all indices in the given dimension are unique
        return len(dim_block_indices) == len(set(dim_block_indices))

    def __str__(self):
        return f"<Condition {self.name} - Dimension {self.dim} - Block {self.divisor}>"


class NotSamePar(NotSameCondition):

    def __init__(self, feature_map):
        super().__init__(f"not_same_par_{feature_map['par']}", feature_map)

    def check_condition(self, comb, params=None, custom_partitions=None):
        pars_list = [get_var_partition_idx(var,custom_partitions) for var in comb]
        specific_par_indices = [pars[self.par] for pars in pars_list]

        # Check if all indices in the given dimension are unique
        return len(specific_par_indices) == len(set(specific_par_indices))


class NotSameName(NotSameCondition):

    def __init__(self, feature_map):
        super().__init__(f"not_same_name", feature_map)

    def check_condition(self, comb, params=None, custom_partitions=None):
        var_names = [get_var_name(var) for var in comb]

        # Check if all indices in the given dimension are unique
        return len(var_names) == len(set(var_names))

class DimDiff(DiffCondition):

    def __init__(self, feature_map, categ, condition=None):
        super().__init__(f"dim_{feature_map['dim']}_diff", feature_map, categ, condition=condition)

    def check_condition(self, comb, params=None, custom_partitions=None):
        assert isinstance(params, dict)
        dims_list = [get_var_dims(var) for var in comb]
        specific_dim_indices = [dims[self.dim] for dims in dims_list]

        # Calculate the average difference between indices
        avg_diff = get_symbolic_parameter(average_difference(specific_dim_indices), params)

        # Check condition
        return any(a in self.allowed_diff for a in avg_diff)


class BlockDiff(DiffCondition):

    def __init__(self, feature_map, categ, condition=None):
        assert "divisor" in feature_map.keys(), "BlockDiff condition attempted to be created without a divisor parameter"
        super().__init__(f"dim_{feature_map['dim']}_block_{feature_map['divisor']}_diff", feature_map, categ, condition=condition)

    def check_condition(self, comb, params=None, custom_partitions=None):
        assert isinstance(params, dict)
        dims_list = [get_var_dims(var) for var in comb]
        specific_dim_indices = [dims[self.dim] for dims in dims_list]
        block_size = params[self.divisor]
        dim_block_indices = [dim // block_size for dim in specific_dim_indices]

        # Calculate the average difference between block indices
        avg_diff = get_symbolic_parameter(average_difference(dim_block_indices), params)

        # Check condition
        return any(a in self.allowed_diff for a in avg_diff)

    def __str__(self):
        return f"<Condition {self.name} - Dimension {self.dim} - Block {self.divisor} - Allowed diff: {self.allowed_diff} - Original Condition: {self.condition}>"

class ParDiff(DiffCondition):

    def __init__(self, feature_map, categ, condition=None):
        super().__init__(f"par_{feature_map['par']}_diff", feature_map, categ, condition=condition)

    def check_condition(self, comb, params=None, custom_partitions=None):
        assert isinstance(params, dict)
        pars_list = [get_var_partition_idx(var,custom_partitions) for var in comb]
        specific_par_indices = [pars[self.par] for pars in pars_list]

        # Calculate the average difference between indices
        avg_diff = get_symbolic_parameter(average_difference(specific_par_indices), params)

        # Check condition
        return any(a in self.allowed_diff for a in avg_diff)
