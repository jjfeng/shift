<h1 align="center"> SHIFT: Subgroup-scanning Hierarchical Inference Framework for performance drifT </h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue">  
</p> 

SHIFT is a diagnostic framework for performance drifts in ML models. It detects the source of the drift by finding subgroups where performance differs due to covariate or outcome shifts. Outputs are a set of hypothesis test results which can guide actions to improve models.

Paper: Singh, Harvineet, Fan Xia, Alexej Gossmann, Andrew Chuang, Julian C. Hong, and Jean Feng. 2025. “‘Who Experiences Large Model Decay and Why?’ A Hierarchical Framework for Diagnosing Heterogeneous Performance Drift.” In Forty-Second International Conference on Machine Learning. [https://openreview.net/forum?id=QtbyoRxyNx](paper)

## Installation instructions
Install required packages by running `pip install -r requirements.txt`

Install `torch-two-sample` library for MMD methods from the repo [https://github.com/josipd/torch-two-sample](https://github.com/josipd/torch-two-sample). It may require installing `cython` from `pip install cython`.

Install `feature-shift` detection library for Score method from the repo [https://github.com/inouye-lab/feature-shift](https://github.com/inouye-lab/feature-shift).

Install `folktables` library for ACS data from the repo [https://github.com/socialfoundations/folktables](https://github.com/socialfoundations/folktables).

Install `R` language for the `tevims` method and copy the repo [https://github.com/ohines/tevims/tree/main](https://github.com/ohines/tevims/tree/main) into `tevims` folder in `src`.

## Reproducing experiments
We use `nestly` and `scons` framework to specify experiments.

Following commands should be run from `src` folder.

1. To run simulations for Setup 1a and 1b, run `scons simulation_agg` where `simulation_agg` folder has the [sconscript](src/simulation_agg/sconscript) file that specifies the experiment setup.
2. For Setup 2 and 3, run `scons simulation_comparators`.
3. For ACS data, run notebook [`src/prepare_acs.ipynb`](src/prepare_acs.ipynb) to generate data files. Run `scons casestudy`.
