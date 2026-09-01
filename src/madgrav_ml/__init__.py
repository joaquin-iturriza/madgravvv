"""madgrav_ml — retrainable ML front end and matched-FAR evaluation harness for MADGRAV.

The upstream search (`ginguglia/MADGRAV`, arXiv:2511.13154) ships frozen weights but
not the training or calibration code. This package reimplements that front end so it
can be retrained, and wraps it in the evaluation protocol the field accepts: detection
efficiency and sensitive volume at a fixed false-alarm rate, measured against
time-slide background pushed through the *same* selection as the foreground.

Layout mirrors `Foundational_Amplitudes`: a Hydra-driven `run.py` entry point over a
`BaseExperiment` harness (`init_physics -> init_data -> init_model -> train ->
evaluate -> plot`), with models, data and evaluation as separate subpackages.
"""

__version__ = "0.1.0"
