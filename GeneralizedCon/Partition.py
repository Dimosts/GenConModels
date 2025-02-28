import enum

from utils.generic_utils import get_var_dims, get_var_name


class PartitionType(enum.Enum):
    var_name = "var_name"
    dim = "dim"
    full = "full"
    block = "block"
    custom = "custom"


class Partition:

    def __init__(self, par_type):

        self._type = par_type

    def generate_partitions(self, var_groups):
        raise NotImplementedError("Attempt to call generate_partitions in generic Partition class")

    def __str__(self):
        return "Generic Partition object - Never use this"
        #return f"Partition of type {self.type}, on relation {self.relation} with constant {self.cnst}, for index {self.index}"

    def __repr__(self):
        return self.__str__()


class NamePartition(Partition):

    def __init__(self):
        super().__init__(PartitionType.var_name)

    def generate_partitions(self, var_groups):
        new_groups = []
        # Iterate over existing groups and split them further
        for group in var_groups:
            dim_groups = {}
            for var in group:
                var_name = get_var_name(var)
                if var_name not in dim_groups:
                    dim_groups[var_name] = []
                dim_groups[var_name].append(var)
            new_groups.extend(dim_groups.values())
        return new_groups


class DimPartition(Partition):

    def __init__(self, dim_index=None):
        super().__init__(PartitionType.dim)
        self.dim_index = dim_index

    def set_dim_index(self, dim_index):
        self.dim_index = dim_index

    def get_dim_index(self):
        return self.dim_index

    def generate_partitions(self, var_groups):
        new_groups = []
        dim = self.dim_index
        # Iterate over existing groups and split them further
        for group in var_groups:
            dim_groups = {}
            for var in group:
                var_dims = get_var_dims(var)
                key = var_dims[dim]
                if key not in dim_groups:
                    dim_groups[key] = []
                dim_groups[key].append(var)
            new_groups.extend(dim_groups.values())
        return new_groups

    def __str__(self):
        return f"Partition on dimension {self.dim_index}"


class BlockPartition(Partition):

    def __init__(self, dim_index=None, block_divisor=None):
        super().__init__(PartitionType.dim)
        self.dim_index = dim_index
        self.block_divisor = block_divisor

    def set_dim_index(self, dim_index):
        self.dim_index = dim_index

    def get_dim_index(self):
        return self.dim_index

    def set_block_divisor(self, block_divisor):
        self.block_divisor = block_divisor

    def get_block_divisor(self):
        return self.block_divisor

    def generate_partitions(self, var_groups, params=None):
        new_groups = []
        dim = self.dim_index
        # Iterate over existing groups and split them further
        for group in var_groups:
            dim_groups = {}
            for var in group:
                var_dims = get_var_dims(var)
                key = var_dims[dim] // params[self.block_divisor]
                if key not in dim_groups:
                    dim_groups[key] = []
                dim_groups[key].append(var)
            new_groups.extend(dim_groups.values())

        return new_groups

    def __str__(self):
        return f"Partition on block {self.block_divisor} of dimension {self.dim_index}"

class CustomPartition(Partition):

    def __init__(self, custom_index=None):
        super().__init__(PartitionType.custom)
        self.custom_index = custom_index

    def set_custom_index(self, custom_index):
        self.custom_index = custom_index

    def get_custom_index(self):
        return self.custom_index

    def generate_partitions(self, var_groups, custom_partition=None):
        assert len(var_groups) == 1, "custom partitioning needs to be applied first"
        new_groups = []
        index = self.custom_index
        key = None
        # Iterate over existing groups and split them further
        for group in var_groups:
            custom_groups = {}
            for var in group:
                for idx, par in enumerate(custom_partition[index]):
                    if var in set(par.flatten()):
                        key = idx
                        break
                if key is None:
                    raise Exception("Variable was not found in any group!!")
                if key not in custom_groups:
                    custom_groups[key] = []
                custom_groups[key].append(var)
            new_groups.extend(custom_groups.values())
        return new_groups

    def __str__(self):
        return f"Custom Partition given, on index {self.custom_index}"
