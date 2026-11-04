"""MDP teammate components.

* features.py  — truth-free cue extraction from Observations (runtime + offline)
* collect.py   — offline data collection from labelled simulator runs
* cue_model.py — fitted cue likelihoods P(cues | class) and the belief update

Ground truth is used ONLY offline, to label the data the likelihoods are fitted
on (the same way transition probabilities are estimated from the simulator). At
runtime the MDP sees only Observations plus the fitted tables.
"""
