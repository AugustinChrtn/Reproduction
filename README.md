This code attempts to replicate the results from

Lopes, M., Lang, T., Toussaint, M., & Oudeyer, P. Y. (2012). Exploration in model-based reinforcement learning by empirically estimating learning progress. Advances in neural information processing systems, 25. Lopes, M., Lang, T., Toussaint, M., & Oudeyer, P. Y. (NIPS 2012).

## Description


* The five agent classes are in `agents.py`
* The environment class is in `lopesworld.py`
* The policy evaluation functions are in `policy_functions.py`
* The play function and the visualisation functions are in `main_functions.py`
* The generation process of the environments, as well as their generation is in `generation_env.py`
* To launch the experiment and get all the result figures, launch `main.py`
* To get all the parameter fitting figures, launch `parameter_fitting.py`
* To install the libraries, use `requirements.txt` or `requirements.yml`
* The folder **Environments** contains the transitions and the rewards of the environments generated
* The folder **Images** contains 2D heatmaps on the different environments
* The folder **Parameter fitting** contains the plots and the data generated with `parameter_fitting.py`
* The folder **Results** contains the plots and the data generated with `main.py`
* You can read the article or the metadata using `article.pdf` and `metadata.yaml`

## Installation 

To clone this repository, use `git clone https://github.com/AugustinChrtn/Reproduction/`

Then, install the required libraries indicated in the `requirements.txt` or `requirements.yml` file.

After these two steps, you can:
* Launch `generation_env.py` to generate the environments.
* Launch `main.py` to get the figures and the data for the article replication.
* Launch `parameter_fitting.py` to get the parameter fitting data and plots on these environments.









