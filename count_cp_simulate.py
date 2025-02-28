import numpy as np
from cpmpy.expressions.utils import all_pairs
from GenAcq import GenAcq, vars_in_rel, generate_expression, get_combinations, combine_sets_distinct
from GeneralizedCon.GenConstraint import GenConstraint
from GeneralizedCon.GenModel import GenModel
from GeneralizedCon.Partition import PartitionType, Partition, DimPartition, CustomPartition
from ProblemInstance import ProblemInstance
from utils.generic_utils import get_var_dims, get_symbolic_parameter_or_val, get_parameter_val_from_name


class CountCP(GenAcq):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_model = GenModel()

    def gen_constraints_in_partition(self, vars, rel, cnst):

        arity = vars_in_rel(rel, 'var')

        if arity == 2:
            for v1, v2 in all_pairs(vars):
                c, _ = generate_expression(rel, [v1, v2], list(cnst))
                yield c
        elif arity == 4:
            combinations = get_combinations(vars, 2)
            result_combinations = combine_sets_distinct(combinations, combinations)
            for ((v1, v2), (v3, v4)) in result_combinations:
                c, _ = generate_expression(rel, [v1, v2, v3, v4])
                yield c

    def learn(self):

        print("finding partitions in countcp")

        input_instance = self.get_input_instances()[0]
        assert isinstance(input_instance, ProblemInstance)

        param_keys = input_instance.get_params().keys()
        dim_lengths = input_instance.get_dim_lengths()
        ndims = max(len(dim_lengths[i]) for i in range(len(dim_lengths)))
        partitions_len = len(input_instance.get_custom_partitions())

        for rel in self._lang:

            constants = vars_in_rel(rel, 'const')
            if constants > 0:
                print(param_keys)
                consts = [[param] * constants for param in param_keys]
            else:
                consts = [()]

            for const in consts:

                # default all_vars partition
                all_vars = True
                for instance in self.get_input_instances():
                    assert isinstance(instance,ProblemInstance)
                    setCin = set(instance.get_constraints())
                    vars = instance.get_variables()
                    params = instance.get_params()
                    const_val = [params[ck] for ck in const]
                    for c in self.gen_constraints_in_partition(vars, rel, const_val):
                        if all([not c in setCin]):
                            all_vars = False
                            break

                if all_vars:
                    gcon = GenConstraint()
                    gcon.set_lang([rel])
                    self.gen_model.add_gen_constraint(gcon)
                    continue

                # default dimension_same partition --- for each dimension
                for dim in range(ndims):
                    dimension_same = True
                    for instance in self.get_input_instances():
                        if not dimension_same:
                            break
                        setCin = set(instance.get_constraints())
                        vars = instance.get_variables()
                        params = instance.get_params()
                        const_val = [params[ck] for ck in const]
                        partitions = [[ var for var in vars if get_var_dims(var)[dim] == i ] for i in range(dim_lengths[0][dim])]
                        for par in partitions:
                            for c in self.gen_constraints_in_partition(par, rel, const_val):
                                if all([not c in setCin]):
                                    dimension_same = False
                                    break
                            if not dimension_same:
                                break

                    if dimension_same:
                        gcon = GenConstraint()
                        gcon.set_lang([rel])
                        gcon.set_constants(const)
                        gcon.set_partitions([DimPartition(dim)])
                        self.gen_model.add_gen_constraint(gcon)

                # custom partitions --- for each type
                for p in range(partitions_len):
                    custom_par = True
                    for instance in self.get_input_instances():
                        if not custom_par:
                            break
                        assert isinstance(instance, ProblemInstance)
                        setCin = set(instance.get_constraints())
                        params = instance.get_params()
                        const_val = [params[ck] for ck in const]
                        partitions = instance.get_custom_partitions()
                        partitions = partitions[p]
                        for par in partitions:
                            for c in self.gen_constraints_in_partition(par.flatten(), rel, const_val):
                                if all([not c in setCin]):
                                    custom_par = False
                                    break
                            if not custom_par:
                                break

                        if custom_par:
                            gcon = GenConstraint()
                            gcon.set_lang([rel])
                            gcon.set_constants(const)
                            gcon.set_partitions([CustomPartition(p)])
                            self.gen_model.add_gen_constraint(gcon)

    def generalize(self):
        instance = self.get_target_instance()
        assert isinstance(instance,ProblemInstance)
        C = self.gen_model.generate_ground_model(self.get_target_instance())
        print("Finished generalizing from countcp")
        C_L = list(dict.fromkeys(C))
        return C_L

    def noisify(self, p, mode):

        for instance in self.get_input_instances():
            assert isinstance(instance, ProblemInstance)
            instance.noisify_constraints(p, mode)


