# End-to-End Learning for Stochastic Optimization: A Bayesian Perspective

## Introduction
This repository contains the code for the experiments presented in the paper _End-to-End Learning for Stochastic Optimization: A Bayesian Perspective_, accepted as a poster at ICML 2023. The paper proposes a general framework to model end-to-end learning approaches for stochastic optimization problems and analyze their behavior. It provides a rigorous Bayesian interpretation for a popular existing end-to-end learning approach, and formulates end-to-end learning algorithmic counterparts to empirical risk minimization and distributionally robust optimization - two standard stochastic optimization methodologies. The experiments illustrate the theoretical results using a synthetic newsvendor problem and an economic energy dispatch problem.

[Paper Link](PLACEHOLDER) | [ICML 2023](https://icml.cc)

## Dependencies and Installation
The code is written in Python. All required packages can be found in the `requirements.txt` file, we use Python 3.9.12 for our experiments. To install the dependencies, run the following command:

```
pip install -r requirements.txt
```


## Usage
Instructions on how to run the code for each experiment can be found in the corresponding subfolders.

## Folder Structure
- `Energy Dispatch`: the code for the energy dispatch experiment
- `Gradient Projection`: the code for the gradient projection experiment
- `Mean Estimation`: the code for the mean estimation experiment
- `Newsvendor experiment`: the code for the newsvendor experiment

## Citation
If you use our code or findings in your research, please cite our paper as follows:

```
@inproceedings{AAA,
    author = {Rychener, Yves and Kuhn, Daniel and Sutter, Tobias},
    title = {End-to-End Learning for Stochastic Optimization: A Bayesian Perspective},
    booktitle = {International Conference on Machine Learning},
    pages={PLACEHOLDER},
    year = {2023},
}
```

## Contact
For questions or support, please contact [Yves Rychener](mailto:yves.rychener@epfl.ch).

## License
This project is licensed under the [MIT License](LICENSE).
