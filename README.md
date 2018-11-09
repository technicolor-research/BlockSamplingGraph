# Structured sampling and fast reconstruction of smooth graph signals
<img src="./pics/fig.png" width="300">
This code is provided to test the sampling procedure presented in:

[1] G. Puy and P. Pérez, "Structured sampling and fast reconstruction of smooth graph signals", Information and Inference: A Journal of the IMA, 2018.

The code is divided into two parts. The matlab code permits to reproduce the experiments in Section 5.1 of [1]. The python code permits one to conduct samplings and reconstructions similar to those appearing in Section 5.2  of [1].

Author: Gilles Puy


## I - Matlab code in subfloder matlab/


This part of the code permits one to reproduce the experiments in Section 5.1 of [1].


* Pre-requisites:

	1. Download and install the Graph Signal Processing Toolbox (GSPBox) available at: https://epfl-lts2.github.io/gspbox-html/.

	2. Download and install the Graph Sampling Box available at: http://grsamplingbox.gforge.inria.fr/.


* Usage:

In matlab, change directory to the subfolder 'matlab/' of the present package.

Launch the script
	main.m 

This script permits to compute the probability that the lower RIP constant is less than 0.995 as a function of the number of sampled groups, for the chosen graph and sampling distribution.

Please refer to the comments in this script to change the graph, the bandlimit, or the sampling distribution.


## II - Python code in subfolder python/

This part of the code permits one to conduct samplings and reconstructions similar to those conducted in Section 5.2  of [1].


* Pre-requisites:

	1. The code was implemented and tested using Python 3.6 on Windows 7.

	2. The following packages are needed:
		* numpy
		* scipy
		* opencv
		* sklearn
		* skimage
		* matplotlib
		* multiprocessing

	3. Download the semantic segmentation dataset of Li et al., ICME, 2013 (link below). Place it in the subfolder 'data/' of this package. http://www.ntu.edu.sg/home/asjfcai/Benchmark_Website/benchmark_index.html.


* Usage:

In a terminal, change directory to the subfolder 'python/' of this package and type:
	python main.py [sampling_distribution]

Three choices are possible for [sampling_distribution]:
	- uniform
	- estimated_fro
	- estimated_spec

For example, typing
	python main.py uniform
will launch simulation using a uniform distribution for sampling the superpixels.

The sampling distribution estimated_fro corresponds to the one presented in Section 3.2 and denoted $\bar{q}$ in [1].

The sampling distribution estimated_spec corresponds to the one presented in Section 3.1 and denoted $\bar{p}$ in [1]. 

Note that estimating the distribution estimated_spec is computationally expensive. By default, this estimation is done using parallel computing with nb_cpu=6 given as option of the function gt.estimate_group_cum_coh_spec in the main script. You can adapt this value depending on the configuration of your computer. To disable parallel computing, set nb_cpu=1.

#### License 
by downloading this program, you commit to comply with the license as stated in the LICENSE.md file.







