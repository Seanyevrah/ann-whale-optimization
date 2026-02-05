# Artificial Neural Network Optimization using Whale Optimization Algorithm (WOA)

This repository contains the implementation and analysis of **Laboratory Activity 4**, focusing on the application of Swarm Intelligence (SI) to enhance Artificial Neural Network (ANN) performance. Specifically, it utilizes the **Whale Optimization Algorithm (WOA)** to automate and optimize the selection of neural network hyperparameters for binary classification.

## Overview

Traditional gradient-based optimization methods often suffer from convergence to local minima and high computational demands. This project explores a nature-inspired alternative: the **Whale Optimization Algorithm**, which mimics the bubble-net hunting behavior of humpback whales.

By treating neural network weights as "prey" and search agents as "whales," the algorithm effectively balances global exploration and local exploitation to find optimal parameters without relying on gradient information.

---

## Core Objectives

* **Implement WOA:** Develop a gradient-free framework to optimize ANN connection weights.


* **Dataset Application:** Apply the WOA-ANN framework to the **Banknote Authentication Dataset** from the UCI Machine Learning Repository.


* **Hyperparameter Analysis:** Investigate the impact of hidden layer size, population size, and regularization () on model performance.


* 
**Performance Evaluation:** Measure success through convergence stability, training accuracy, and test accuracy.



---

## Technical Stack

* **Development Environment:** GNU Octave.


* **Optimization Strategy:** Whale Optimization Algorithm (WOA).


* **Architecture:** Fully connected Artificial Neural Network (ANN).


* **Data Processing:** Feature normalization (zero mean and unit variance).



---

## Mathematical Foundations of WOA

The optimization process mimics three primary biological behaviors:

1. **Encircling Prey:** Search agents update their positions based on the current best solution.


2. **Bubble-net Attacking (Exploitation):** A spiral-shaped movement modeled to simulate the actual hunting maneuver.


3. **Search for Prey (Exploration):** Random exploration of the search space to maintain diversity and avoid local optima.



---

## Results and Discussion

The framework demonstrated high classification accuracy and stable convergence.

| Configuration | Best Cost | Training Accuracy | Test Accuracy |
| --- | --- | --- | --- |
| Default (5 Hidden, 100 Pop) | 0.105828 | 98.96% | 97.09% |
| Smaller Population (50 Pop) | 0.180814 | 98.85% | 100.00% |
| Alternative Split (80-20) | 0.079336 | 99.18% | 99.27% |

Data summarized from experimental test cases.

---

## Repository Structure

* `main.m`: The primary script to load data, initialize parameters, and run the optimization.
* `woa_nn.m`: Core implementation of the Whale Optimization Algorithm for Neural Networks.
* `nnCostFunction.m`: Computes the cost for a two-layer neural network with regularization.
* `featureNormalize.m`: Standardizes features for improved training stability.
* `predict.m`: Predicts labels using optimized weights.
