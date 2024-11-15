# The Uniformly Rotated Mondrian Kernel

Calvin Osborne and Eliza O'Reilly

The scripts provided here implement experiments from this paper. The scripts `experiment_1_limiting_kernel_convergence`, `experiment_2_ground_truth_lifetime`, `experiment_3_experimental_dataset_features`, and `experiment_4_experimental_dataset_time` are intended to be directly runnable. The file `main.py` will sequentially run all four experiments. 

A fixed seed is used for all four experiments and can be found or changed in `config.py` (by default, and to generate our figures, use `seed = 0`).

For more information about liscensing, including the original license of the modified code framework, see `NOTICE.md`.

### Requirements

Python Packages: `matplotlib`, `numpy`, `scipy`, `sklearn`, `sys`, `time`.

Version Information:
* `python`: 3.12.6
* `matplotlib`: 3.9.2
* `numpy`: 2.1.1
* `scipy`: 1.14.1
* `sklearn`: 1.5.2

The CPU dataset `cpu.mat` can be download \[[here](https://www.dropbox.com/scl/fi/f0zb1sllf5147xyr4fxgu/cpu.mat?rlkey=xnpbd8id9v131nebtvj3xqhke&st=ju2vdact&dl=0)\].
