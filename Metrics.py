from utils.generic_utils import compare_vars, unravel

class Metrics:

    def __init__(self, C_L=None, C_T=None):

        if C_L is None:
            C_L = []
        if C_T is None:
            C_T = []

        self._C_L = C_L
        self._C_T = C_T

        self.generalized_vars = []
        self.target_vars = []

        self.found_vars = None
        self.wrong_vars = None
        self.missing_vars = None

        self.found_constraints = None
        self.true_pos = None
        self.false_pos = None
        self.false_neg = None

        self.total_time = None
        self.training_time = None
        self.prediction_time = None

    # Getter for C_L
    def get_C_L(self):
        return self._C_L

    # Setter for C_L
    def set_C_L(self, C_L):
        self._C_L = C_L

    # Getter for C_T
    def get_C_T(self):
        return self._C_T

    # Setter for C_T
    def set_C_T(self, C_T):
        self._C_T = C_T

    def set_times(self, training_time, prediction_time):
        self.training_time = training_time
        self.prediction_time = prediction_time
        self.total_time = training_time + prediction_time

    def evaluate_vars(self):

        self.found_vars = 0
        self.wrong_vars = 0

        lst = []
        unravel(self.generalized_vars, lst)
        self.generalized_vars = lst

        for var1 in self.generalized_vars:

            # check if var1 is in target vars
            if any([compare_vars(var1, var2) for var2 in self.target_vars]):
                self.found_vars +=1
            else:
                self.wrong_vars +=1

        self.missing_vars = len(self.target_vars) - self.found_vars
        print(f"found_vars: {self.found_vars}")
        print(f"wrong_vars: {self.wrong_vars}")
        print(f"missing_vars: {self.missing_vars}")

    def evaluate(self):

        print("evaluating")
        setCT = set(self._C_T)
        setCL = set(self._C_L)

        self.found_constraints = len(self._C_L)
        self.true_pos = 0
        self.false_pos = 0

        for c in self._C_L:
            #if any([compare_constraints(c, c2) for c2 in setCT]):
            if c in setCT:
                self.true_pos += 1
            else:
                self.false_pos += 1

        self.false_neg = len(self._C_T) - (len(self._C_L) - self.false_pos)

        print(f"found: {self.found_constraints}, true positive: {self.true_pos}, false positive: {self.false_pos}, "
              f"false_negative: {self.false_neg}, C_T: {len(self._C_T)}")
