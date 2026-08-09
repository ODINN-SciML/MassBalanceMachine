import itertools


def flatten_dict(d, parent_key="", sep="."):
    result = {}
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            result.update(flatten_dict(value, new_key, sep))
        else:
            if isinstance(value, list):
                if isinstance(value[0], (tuple, list)):
                    result[new_key] = tuple(tuple(v) for v in value)
                else:
                    result[new_key] = tuple(value)
            else:
                result[new_key] = value
    return result


def recursive_update(target, source):
    for key in source:
        assert key in target.keys()
        if isinstance(target[key], dict):
            recursive_update(target[key], source[key])
        else:
            target[key] = source[key]


def recursive_update_from_flat(target, source, sep="."):
    for flat_key, value in source.items():
        parts = flat_key.split(sep)
        _set_recursive(target, parts, value)


def _set_recursive(current, parts, value):
    key = parts[0]
    if len(parts) == 1:
        current[key] = value
        return
    if key not in current or not isinstance(current[key], dict):
        current[key] = {}
    _set_recursive(current[key], parts[1:], value)


def canonicalize(x):
    """Convert params into a stable, comparable representation."""
    if isinstance(x, Mapping):
        return tuple(sorted((str(k), canonicalize(v)) for k, v in x.items()))
    if isinstance(x, tuple):
        return tuple(canonicalize(v) for v in x)
    if isinstance(x, list):
        return tuple(canonicalize(v) for v in x)
    if isinstance(x, float):
        if math.isnan(x):
            return "__nan__"
        return x
    return x


def canonicalize_raw(params: dict):
    return tuple(sorted((k, v) for k, v in params.items()))


def get_next_untested_params(study, search_space):
    """
    Returns the next parameter combination not yet COMPLETE or RUNNING,
    without allocating a trial first.
    """
    from optuna.trial import TrialState

    # Get the keys of the search space to build a set of parameters in which the keys are always in the same order
    keys = list(search_space.keys())

    # Get all already-seen param combinations which are either complete or running
    # We exclude failed combinations because we want to test them again
    seen = set()
    for t in study.get_trials(
        deepcopy=False,
        states=(TrialState.COMPLETE, TrialState.RUNNING),  # , TrialState.FAIL
    ):
        seen.add(canonicalize_raw(t.params))  # t.params stores raw categoricals

    # Enumerate the full grid
    values = [search_space[k] for k in keys]  # already tuples of categoricals

    for combo in itertools.product(*values):
        candidate_raw = dict(zip(keys, combo))
        if canonicalize_raw(candidate_raw) not in seen:
            return candidate_raw

    return None  # grid exhausted
