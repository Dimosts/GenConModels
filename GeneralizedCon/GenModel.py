import numpy as np

from GeneralizedCon.GenConstraint import GenConstraint
from utils.generic_utils import parse_dt_rule, get_scope


class GenModel:

    def __init__(self, gen_constraints=None, lang=None, params=None, rules=None, feature_mapping=None, feature_categories=None):
        if gen_constraints is None:
            gen_constraints = []
        self.gen_constraints = gen_constraints

        if lang is None:
            lang = []
        self.lang = lang

        if params is None:
            params = []
        self. params = params

        self.rules = rules
        self.feature_mapping = feature_mapping
        self.feature_categories = feature_categories

    def set_gen_constraints(self, gen_constraints):
        self.gen_constraints = gen_constraints

    def add_gen_constraint(self, gen_constraint):
        self.gen_constraints.append(gen_constraint)

    def get_gen_constraints(self):
        return self.gen_constraints

    def set_lang(self, lang):
        self.lang = lang

    def set_params(self, params):
        self.params = params

    def get_lang(self):
        return self.lang

    def get_params(self):
        return self.params

    def set_rules(self, rules):
        self.rules = rules

    def get_rules(self):
        return self.rules

    def set_feature_mapping(self, feature_mapping):
        self.feature_mapping = feature_mapping

    def get_feature_mapping(self):
        return self.feature_mapping

    def set_feature_categories(self, feature_categories):
        self.feature_categories = feature_categories

    def get_feature_categories(self):
        return self.feature_categories

    def create_gen_model_from_cn2(self):
        assert self.rules is not None, "Rules must be initialized to create generalized model"

        for rule in self.rules:
            c = GenConstraint(lang=self.lang, params=self.params)
            c.create_genconstraint_from_cn2_rule(rule, self.feature_mapping, self.feature_categories)
            self.gen_constraints.append(c)

    def create_gen_model_from_dt(self, feature_names, encoders):
        assert self.rules is not None, "Rules must be initialized to create generalized model"

        for rule in self.rules:
            c = GenConstraint(lang=self.lang, params=self.params)
            c.create_genconstraint_from_dt_rule(parse_dt_rule(rule), self.feature_mapping, self.feature_categories, feature_names, encoders)
            self.gen_constraints.append(c)

    def generate_ground_model(self, instance):
        # Simplify ground model generation
        return [
            constraint 
            for gen_constraint in self.gen_constraints
            for constraint in gen_constraint.generate_ground_constraints(instance)
        ]

