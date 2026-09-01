"""Experiments. Each is one run of the harness, selected by `exp_type` in the config.

Every experiment subclasses `BaseExperiment` and therefore inherits the fold guard,
the C2 parameter check, the run-directory contract and the record. Deviating from any
of those requires a stated reason in the module docstring.
"""
