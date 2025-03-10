Problem:

$2025 = 45^2 = (1+2+\ldots+9)^2 = 1^3 + 2^3 + \ldots + 9^3$

*How many combinations of 1 square of size 1, 2 squares of size 2, 3 squares of size 3, ..., 8 squares of size 8, 9 squares of size 9 exist such that they fill the square of size 45 with no intersection?*

This repository contains 3 main files:

1. cuda9_3.py - main file for computing the combinations
2. process9.py - auxillary file for processing results
3. plot.py - auxillary file for visualization

Use requirements.txt file to install pip libraries (preferably in a virtual environment).

Computation requires Nvidia GPU with CUDA installed.
For a quick test set N = int32(8) and SIZE = int32(36) in cuda9_3.py. You should get 2332 after running cuda9_3.py and process9.py.

You can see a few examples of combinations of SIZE 45 in the file 100examples.zip.
