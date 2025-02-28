from cpmpy import *
from cpmpy.expressions.core import Comparison
from cpmpy.expressions.utils import all_pairs
from sklearn.preprocessing import LabelEncoder

from ProblemInstance import ProblemInstance
from utils.generic_utils import isPartition, vars_in_rel, generate_expression, get_combinations, combine_sets_distinct
from GeneralizedCon.Partition import *
from GeneralizedCon.Condition import *

class GenConstraint:

    def __init__(self, lang=None, params=None, relations=None, constants=None, partition=None, seq_conditions=None):
        # Use list comprehension for cleaner initialization
        self.lang = lang or []
        self.params = params or {}
        self.constants = constants or (list(params.keys()) if params else [])
        self.relations = relations
        self.partition = partition or []
        self.seq_conditions = seq_conditions or []
        
        if len(self.seq_conditions) > 1:
            self.combine_seq_conditions()

    def set_lang(self, lang):
        self.lang = lang

    def set_params(self, params):
        self.params = params

    def set_relations(self, relations):
        self.relations = relations

    def set_constants(self, constants):
        self.constants = constants

    def set_partitions(self, partition):
        self.partition = partition

    def set_seq_conditions(self, seq_conditions):
        self.seq_conditions = seq_conditions
        self.combine_seq_conditions()

    def combine_seq_conditions(self):

        combined_conditions = {}

        for condition in self.seq_conditions:
            if not isinstance(condition, DiffCondition):
                continue
            if condition.name in combined_conditions:
                combined_conditions[condition.name].allowed_diff = list(
                    set(combined_conditions[condition.name].allowed_diff) & set(condition.allowed_diff)
                )
            else:
                combined_conditions[condition.name] = condition

        self.seq_conditions = list(combined_conditions.values())

    def get_relations(self):
        return self.relations

    def get_partition(self):
        return self.partition

    def get_constants(self):
        return self.constants

    def get_seq_conditions(self):
        return self.seq_conditions

    def get_lang(self):
        return self.lang

    def get_params(self):
        return self.params

    def create_genconstraint_from_cn2_rule(self, rule, feature_mapping, feature_categories):
        """
        :param rules: Rules given from a rule based classifier
        :return:
        """
        # assert isinstance(rule,classification.)
        self.relations = []
        self.partition = []
        self.seq_conditions = []

        for selector in rule.selectors:
            feature_index = selector.column
            op = selector.op
            val = selector.value

            feature = feature_mapping[feature_index]
            categories = feature_categories[feature_index]
            assert isinstance(feature, dict)

            if "Partition" in feature.keys():
                if isPartition(op, val):
                    if feature["Partition"] == "dim":
                        self.partition.append(DimPartition(feature["dim"]))
                    elif feature["Partition"] == "block":
                        self.partition.append(BlockPartition(feature["dim"], feature["divisor"]))
                    elif feature["Partition"] == "custom":
                        self.partition.append(CustomPartition(feature["par"]))
                    else:
                        self.partition.append(NamePartition())
                else:
                    if feature["Partition"] == "dim":
                        self.seq_conditions.append(NotSameDim(feature))
                    elif feature["Partition"] == "block":
                        self.seq_conditions.append(NotSameBlock(feature))
                    elif feature["Partition"] == "custom":
                        self.seq_conditions.append(NotSamePar(feature))
                    else:
                        self.seq_conditions.append(NotSameName(feature))

            elif "Sequence" in feature.keys():
                allowed_categ = [c for idx, c in enumerate(categories) if Comparison(op,idx,int(val)).value()]
                if feature["Sequence"] == "dim":
                    self.seq_conditions.append(DimDiff(feature, allowed_categ))
                elif feature["Sequence"] == "block":
                    self.seq_conditions.append(BlockDiff(feature, allowed_categ))
                else:
                    self.seq_conditions.append(ParDiff(feature, allowed_categ))
            else:
                allowed_categ = [c for idx, c in enumerate(categories) if Comparison(op, idx, int(val)).value()]
                self.restrict_relations(feature, allowed_categ)
        self.combine_seq_conditions()
        #print(self)

    def create_genconstraint_from_dt_rule(self, rule, feature_mapping, feature_categories, feature_names, label_encoders):
        """
        :param rules: Rules given from a rule based classifier
        :return:
        """
        # assert isinstance(rule,classification.)
        self.relations = []
        self.partition = []
        self.seq_conditions = []

        for condition in rule:
            feature, op, val = condition["feature"], condition["op"], float(condition["value"])
            feature_index = feature_names.index(feature)
            feature = feature_mapping[feature_index]
            categories = feature_categories[feature_index]
            encoder = label_encoders[feature_index]
            assert isinstance(encoder, LabelEncoder)
            assert isinstance(feature, dict)
            if "Partition" in feature.keys():
                if isPartition(op, val):
                    if feature["Partition"] == "dim":
                        self.partition.append(DimPartition(feature["dim"]))
                    elif feature["Partition"] == "block":
                        self.partition.append(BlockPartition(feature["dim"], feature["divisor"]))
                    elif feature["Partition"] == "custom":
                        self.partition.append(CustomPartition(feature["par"]))
                    else:
                        self.partition.append(NamePartition())
                else:
                    if feature["Partition"] == "dim":
                        self.seq_conditions.append(NotSameDim(feature))
                    elif feature["Partition"] == "block":
                        self.seq_conditions.append(NotSameBlock(feature))
                    elif feature["Partition"] == "custom":
                        self.seq_conditions.append(NotSamePar(feature))
                    else:
                        self.seq_conditions.append(NotSameName(feature))

            elif "Sequence" in feature.keys():
                allowed_categ = [c for c in categories if Comparison(op,encoder.transform([c])[0],val).value()]
                if feature["Sequence"] == "dim":
                    self.seq_conditions.append(DimDiff(feature, allowed_categ, condition=condition))
                elif feature["Sequence"] == "block":
                    self.seq_conditions.append(BlockDiff(feature, allowed_categ, condition=condition))
                else:
                    self.seq_conditions.append(ParDiff(feature, allowed_categ, condition=condition))
            else:
                allowed_categ = [c for c in categories if Comparison(op,encoder.transform([c])[0],val).value()]
                self.restrict_relations(feature, allowed_categ)
        self.combine_seq_conditions()
        #print(self)

    # function to generate directly the constraints for the given instance, based on the generalized constraint and the
    # instance parameters
    def generate_ground_constraints(self, instance: ProblemInstance):

        print("Generating ground constraints -----")
        partitions = self.generate_partitions(instance)
        lang, const_params = self.lang, self.constants
        params = instance.get_params()
        custom_partitions = instance.get_custom_partitions()
        C = []

        for r in lang:
            C2 = []
            arity = vars_in_rel(r, 'var')
            constants = vars_in_rel(r, 'const')
            if constants > 0:
                consts = [[param] * constants for param in [params[const] for const in const_params]]
                consts = [list(x) for x in dict.fromkeys([tuple(sublist) for sublist in consts])]
            else:
                consts = [()]
            for const in consts:
                for partition in partitions:
                    if arity == 2:
                        pairs = all_pairs(partition)
                        pairs = [pair for pair in pairs if self.check_conditions(pair, params, custom_partitions)]
                        for v1, v2 in pairs:
                            c, _ = generate_expression(r, [v1, v2], list(const))
                            C2.append(c)
                    elif arity == 4:
                        combinations = get_combinations(partition, 2)
                        result_combinations = combine_sets_distinct(combinations, combinations)
                        for ((v1, v2), (v3, v4)) in result_combinations:
                            c, _ = generate_expression(r, [v1, v2, v3, v4])
                            C2.append(c)
            C.extend(C2)

        print(f"Generated {len(C)} constraints")
        return C

    def restrict_relations(self, feature, allowed_categ):
        const_params = self.constants
        lang = self.lang

        if feature["Relation"] == "lang":
            lang = [lang[idx] for idx in allowed_categ]
        elif feature["Relation"] == "constant_param":
            const_params = [cnst for cnst in const_params if cnst in allowed_categ]
        elif feature["Relation"] == "has_constant":
            lang = [l for l in lang if (vars_in_rel(l, 'const') > 0) in [eval(c) for c in allowed_categ]]

        self.lang = lang
        self.constants = const_params


    def generate_partitions(self, instance):

        # Start with a single group containing all variables
        groups = [instance.get_variables()]

        for partition in self.partition:
            # Generate partitions and replace the current groups with the newly created subgroups
            if isinstance(partition, DimPartition):
                groups = partition.generate_partitions(groups)
            elif isinstance(partition, BlockPartition):
                dim_lengths = instance.get_dim_lengths()
                params = instance.get_params()
                if all(dim_lengths[i][partition.get_dim_index()] % params[partition.get_block_divisor()] == 0 for i in range(len(dim_lengths))):
                    groups = partition.generate_partitions(var_groups=groups, params=params)
            elif isinstance(partition, CustomPartition):
                groups = partition.generate_partitions(var_groups=groups,
                                                       custom_partition=instance.get_custom_partitions())

        return groups

    def check_conditions(self, comb, params, partitions):
        return all(condition.check_condition(comb, params=params, custom_partitions=partitions) for condition in self.seq_conditions)

    def __str__(self):
        repr = f"Generalized constraint with: \nrelation(s) {self.relations} \nlang: {self.lang} \npartitioning {self.partition} \nsequence conditions: {self.seq_conditions}"
        return repr







