import numpy as np
import cpmpy
from Orange.classification.rules import Selector
from cpmpy import Model
from cpmpy.expressions.utils import is_any_list
from cpmpy.expressions.globalfunctions import Abs
from cpmpy.expressions.core import Expression, Comparison, Operator
from cpmpy.expressions.variables import _NumVarImpl, _BoolVarImpl, _IntVarImpl, NDVarArray, boolvar
from cpmpy.transformations.normalize import toplevel_list
from cpmpy.expressions.utils import is_num
from itertools import combinations
import warnings

from sklearn.tree import _tree
from sklearn.utils import class_weight
import re


## to attach to CPMpy expressions!!
def expr_get_relation(self):
    # flatten and replace
    flatargs = []
    for arg in self.args:
        if isinstance(arg, np.ndarray):
            for a in arg.flat:
                if isinstance(a, _NumVarImpl):
                    flatargs.append("var")
                elif isinstance(a, Expression):
                    flatargs.append(a.get_relation())
                else:
                    flatargs.append("const")
        else:
            if isinstance(arg, _NumVarImpl):
                flatargs.append("var")
            elif isinstance(arg, Expression):
                flatargs.append(arg.get_relation())
            else:
                flatargs.append("const")

    if len(flatargs) > 1:
        flatargs = tuple(flatargs)
    else:
        flatargs = flatargs[0] if not isinstance(flatargs[0], _NumVarImpl) else tuple(flatargs)

    return (self.name, flatargs)


Expression.get_relation = expr_get_relation  # attach to CPMpy expressions


def comp_get_relation(self):
    flatargs = []

    for arg in self.args:
        if isinstance(arg, _NumVarImpl):
            flatargs.append("var")
        elif isinstance(arg, Expression):
            flatargs.append(arg.get_relation())
        else:
            flatargs.append("const")

    return (self.name, (flatargs[0], flatargs[1]))


Comparison.get_relation = comp_get_relation  # attach to CPMpy comparisons


# Operator.get_relation = comp_get_relation  # attach to CPMpy operators

def get_variables_from_constraints(constraints):
    def get_variables(expr):
        if isinstance(expr, _IntVarImpl):
            return [expr]
        elif isinstance(expr, _BoolVarImpl):
            return [expr]
        elif isinstance(expr, np.bool_):
            return []
        elif isinstance(expr, np.int_) or isinstance(expr, int):
            return []
        else:
            # Recursively find variables in all arguments of the expression
            return [var for argument in expr.args for var in get_variables(argument)]

    # Create set to hold unique variables
    variable_set = set()
    for constraint in constraints:
        variable_set.update(get_variables(constraint))

    extract_nums = lambda s: list(map(int, s.name[s.name.index("[") + 1:s.name.index("]")].split(',')))

    variable_list = sorted(variable_set, key=extract_nums)
    return variable_list


def vars_in_rel(t, item):
    if isinstance(t, str):
        return t == item
    elif isinstance(t, tuple):
        return sum(vars_in_rel(i, item) for i in t)
    return 0


def generate_expression(rel, vars, consts=None):
    comparison = {'==', '!=', '<=', '<', '>=', '>'}
    operator = {'and', 'or', '->', 'not', 'sum', 'wsum', 'sub', 'mul', 'div', 'mod', 'pow', '-'}

    assert is_any_list(rel), "the relation given does not contain arguments"
    assert is_any_list(vars), "the argument vars must be a list"
    if not (consts is None):
        assert is_any_list(consts), "the argument consts must be a list or None"
        assert (len(vars) + len(consts)) >= (
                len(rel) - 1), "the arguments given are less than the arguments required for the relation"
    else:
        assert len(vars) >= (len(rel) - 1), "the arguments given are less than the arguments required for the relation:" \
                                            f"vars: {vars}, rel: {rel}, {len(rel)} "

    rel_name = rel[0]
    rel_args = rel[1]

    args = []
    if rel_name in ['abs', '-']:
        if is_any_list(rel_args):
            arg, vars = generate_expression(rel_args, vars, consts)
            args.append(arg)
        elif "var" in rel_args:
            args.append(vars.pop(0))
        elif "const" in rel_args:
            args.append(consts.pop(0))
        else:
            raise Exception("Non implemented error")
    else:
        for arg in rel_args:
            if is_any_list(arg):
                arg, vars = generate_expression(arg, vars, consts)
                args.append(arg)
            elif "var" in arg:
                args.append(vars.pop(0))
            elif "const" in arg:
                args.append(consts.pop(0))
            else:
                raise Exception("Non implemented error")

    if rel_name in comparison:
        expr = Comparison(rel_name, args[0], args[1])
    elif rel_name in operator:
        expr = Operator(rel_name, args)
    elif rel_name == 'abs':
        expr = Abs(args[0])
    else:
        raise Exception(f"expression name not a comparison or operator that are allowed: {rel_name}, relation: {rel}")

    return expr, vars


def unravel(lst, newlist):
    for e in lst:
        if isinstance(e, Expression):
            if isinstance(e, NDVarArray):  # sometimes does not have .name
                unravel(e.flat, newlist)
            else:
                newlist.append(e)  # presumably the most frequent case
        elif isinstance(e, (list, tuple, np.flatiter, np.ndarray)):
            unravel(e, newlist)


def get_combinations(lst, n):
    if is_any_list(lst):
        newlist = []
        unravel(lst, newlist)
        lst = newlist
    return list(combinations(lst, n))


# Combine sets with checking distinct elements
def combine_sets_distinct(set1, set2):
    result = set()
    examined = set()
    for a in set1:
        examined.add(a)
        for b in set2:
            if a is not b and b not in examined:
                # Add tuple of sorted combinations to remove duplicates in different order
                result.add(tuple(sorted((a, b))))
    return result


def combine_sets(set1, set2):
    return [(a, b) for a in set1 for b in set2]


def get_scopes_vars(C):
    return set([x for scope in [get_scope(c) for c in C] for x in scope])


def get_scopes(C):
    return list(set([tuple(get_scope(c)) for c in C]))


def get_scope(constraint):
    if isinstance(constraint, cpmpy.expressions.variables._IntVarImpl):
        return [constraint]
    elif isinstance(constraint, cpmpy.expressions.core.Expression):
        all_variables = []
        for argument in constraint.args:
            if isinstance(argument, cpmpy.expressions.variables._IntVarImpl):
                # non-recursive shortcut
                all_variables.append(argument)
            else:
                all_variables.extend(get_scope(argument))
        return all_variables
    else:
        return []


def get_constant(constraint):
    if isinstance(constraint, cpmpy.expressions.variables._IntVarImpl):
        return []
    elif isinstance(constraint, cpmpy.expressions.core.Expression):
        constants = []
        for argument in constraint.args:
            if not isinstance(argument, cpmpy.expressions.variables._IntVarImpl):
                constants.extend(get_constant(argument))
        return constants
    else:
        return [constraint]


def get_arity(constraint):
    return len(get_scope(constraint))


def get_var_name(var):
    name = re.findall("\[\d+[,\d+]*\]", var.name)
    name = var.name.replace(name[0], '')
    return name


def get_var_ndims(var):
    dims = re.findall("\[\d+[,\d+]*\]", var.name)
    dims_str = "".join(dims)
    ndims = len(re.split(",", dims_str))
    return ndims

def get_var_partition_idx(var, partitions):
    keys = []
    for partition in partitions:
        key = None
        for idx, par in enumerate(partition):
            if var in set(par.flatten()):
                key = idx
                break
        if key is not None:
            keys.append(key)
    return keys


def get_var_dims(var):
    dims = re.findall("\[\d+[,\d+]*\]", var.name)
    dims_str = "".join(dims)
    dims = re.split("[\[\]]", dims_str)[1]
    dims = [int(dim) for dim in re.split(",", dims)]
    return dims


def compare_vars(var1, var2):
    assert isinstance(var1, _NumVarImpl), f"var1 is not a variable: {var1}"
    assert isinstance(var2, _NumVarImpl), f"var2 is not a variable: {var2}"

    # check variables' names
    if var1.name != var2.name:
        return False

    # check variables' bounds
    if var1.get_bounds() != var2.get_bounds():
        return False

    # check variables' number of dimensions
    ndims = get_var_ndims(var1)
    if get_var_ndims(var2) != ndims:
        return False

    # check variables' dimension indices
    dims = get_var_dims(var1)
    for i, dim in enumerate(get_var_dims(var2)):
        if dim != dims[i]:
            return False

    return True


def compare_subexpressions(expr1, expr2):
    assert isinstance(expr1, Expression) or is_num(expr1), f"expr1 is not a expression: {expr1}"
    assert isinstance(expr2, Expression) or is_num(expr2), f"expr2 is not a expression: {expr2}"

    # print(f"compare_subexpressions {expr1}, {expr2}")

    if type(expr1) != type(expr2):
        return False

    if is_num(expr1):
        if expr1 != expr2:
            return False
        return True

    # check expression names
    if expr1.name != expr2.name:
        return False

    # if they are variables, compare them as variables
    if isinstance(expr1, _NumVarImpl) and not compare_vars(expr1, expr2):
        return False
    if not isinstance(expr1, _NumVarImpl) and not is_num(
            expr1):  # and not compare_subexpressions(expr1, expr2):  # if they are expressions, compare them as expressions
        vars1 = expr1.args
        vars2 = expr2.args

        # check amount of variables
        if len(vars1) != len(vars2):
            return False

        # check same variables or subexpressions
        if expr1.name in ["sum", "mul"]:
            for var1 in vars1:
                if not any([compare_subexpressions(var1, var2) for var2 in vars2]):
                    return False
        else:
            for var1, var2 in zip(vars1, vars2):
                if not compare_subexpressions(var1, var2):
                    return False

    return True


def compare_constraints(con1, con2):
    assert isinstance(con1, Expression) and con1.is_bool(), f"con1 is not a constraint: {con1}"
    assert isinstance(con2, Expression) and con2.is_bool(), f"con2 is not a constraint: {con2}"

    flag = True

    # check constraints names
    if con1.name != con2.name:
        flag = False

    # check constraints relations
    if con1.get_relation() != con2.get_relation():
        flag = False

    # Check scope of constraints
    vars1 = con1.args
    vars2 = con2.args

    # check amount of variables
    if len(vars1) != len(vars2):
        flag = False

    # check same variables or subexpressions
    if con1.name in ["!=", "=="]:
        for var1 in vars1:
            if not any([compare_subexpressions(var1, var2) for var2 in vars2]):
                flag = False
    else:
        for var1, var2 in zip(vars1, vars2):
            if not compare_subexpressions(var1, var2):
                flag = False

    if not flag:
        if str(con1) == str(con2):
            raise Exception(f"these 2 constraints are the same and it returns false!! {con1}, {con2}")

    return flag


def isPartition(op, val):
    temp_bool = boolvar()
    c = Comparison(op, temp_bool, val)
    return Model([c, temp_bool]).solve()


def get_divisors(n):
    divisors = list()
    for i in range(2, int(n / 2) + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors


def compute_sample_weights(Y):
    c_w = class_weight.compute_class_weight('balanced', classes=np.unique(Y), y=Y)
    sw = []

    for i in range(len(Y)):
        if Y[i] == False:
            sw.append(c_w[0])
        else:
            sw.append(c_w[1])

    return sw


def average_difference(values):
    if len(values) < 2:
        return 0
    differences = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    return sum(differences) / len(differences)


def evaluate(C_L=None, C_T=None):
    setCT = set(C_T)
    setCL = set(C_L)

    found_constraints = len(C_L)
    true_pos = 0
    false_pos = 0

    for c in C_L:
        if c in setCT:
            true_pos += 1
        else:
            false_pos += 1

    # for c in C_T:
    #    if c not in setCL:
    #        print(f"false negative: {c}")

    false_neg = len(C_T) - (len(C_L) - false_pos)

    print(
        f"found: {found_constraints}, true positive: {true_pos}, false positive: {false_pos}, false_negative: {false_neg}")
    return found_constraints, true_pos, false_pos, false_neg


def get_symbolic_parameter(val, params):
    keys = [key for key, value in params.items() if value == val]
    if keys:
        return keys
    else:
        return ["NaN"]


def get_parameter_val_from_name(param_name, params):
    try:
        return params[param_name]
    except KeyError:
        # return param_name
        raise Exception(f'Parameter name given does not exist!! parameter: {param_name}, parameters: {params}')


def get_symbolic_parameter_or_val(val, params):
    return next((key for key, value in params.items() if value == val), val)


def get_instance_parameter(params, i):
    try:
        return params[i]
    except LookupError:
        raise Exception(f'Instance does not have parameter {i}')


def get_rules_from_dt(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"{name} <= {np.round(threshold, 3)}"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"{name} > {np.round(threshold, 3)}"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = ""

        for p in path[:-1]:
            if rule != "":
                rule += " & "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"{class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
        # rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules


def parse_dt_rule(rule):
    rule = rule.split("then")[0]
    conditions = rule.split("& ")

    pattern = re.compile(r'(\S+)\s*(<=|>=|<|>|==|!=)\s*(\S+)')
    parsed_conditions = []
    for cond in conditions:
        match = pattern.match(cond)
        if match:
            feature, op, value = match.groups()
            parsed_condition = {"feature": feature, "op": op, "value": value}
            parsed_conditions.append(parsed_condition)
        else:
            warnings.warn(Warning('Rule does not match pattern of DT rules - May be an empty rule'))

    return parsed_conditions

# Function to reverse the condition
def reverse_selector(selector):

    op = selector.op
    if '<=' in op:
        selector = selector._replace(op='>')
    elif '>=' in op:
        selector = selector._replace(op='<')
        #selector.op = op.replace('>=', '<')
    elif '<' in op:
        selector = selector._replace(op='>=')
        #selector.op = op.replace('<', '>=')
    elif '>' in op:
        selector = selector._replace(op='<=')
        #selector.op = op.replace('>', '<=')
    elif '==' in op:
        selector = selector._replace(op='!=')
        #selector.op = op.replace('==', '!=')
    elif '!=' in op:
        selector = selector._replace(op='==')
        #selector.op = op.replace('!=', '==')
    else:
        raise Exception(f"Operator in selector not in predefined list {op}")

    return selector

# Function to extract conditions from a rule
def extract_cn2_conditions(rule):
    # Remove the "IF" and "THEN" parts
    rule_str = str(rule)
    conditions_part = rule_str.split("THEN")[0].replace("IF", "").strip()
    return conditions_part