from inspect import signature

from utils import export


@export
def is_valid_policy_function(policy):
    if callable(policy):
        sig = signature(policy)
        if len(sig.parameters) == 1:  # this policy must take 1 parameter (state)
            return True
    return False
