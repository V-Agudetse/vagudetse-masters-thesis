#!/bin/bash

./model_lagged_regression.py 2 DJF 0
./model_lagged_regression.py 2 MAM 0
./model_lagged_regression.py 2 JJA 0
./model_lagged_regression.py 2 SON 0
./model_lagged_regression.py 2 DJF 1
./model_lagged_regression.py 2 SON 1
./model_lagged_regression.py 2 JJA 1
./model_lagged_regression.py 2 MAM 1

./model_lagged_regression.py 3 DJF 0
./model_lagged_regression.py 3 MAM 0
./model_lagged_regression.py 3 JJA 0
./model_lagged_regression.py 3 SON 0
./model_lagged_regression.py 3 DJF 1
./model_lagged_regression.py 3 SON 1
./model_lagged_regression.py 3 JJA 1
./model_lagged_regression.py 3 MAM 1

./obs_lagged_regression.py ERSST DJF 0
./obs_lagged_regression.py ERSST MAM 0
./obs_lagged_regression.py ERSST JJA 0
./obs_lagged_regression.py ERSST SON 0
./obs_lagged_regression.py ERSST DJF 1
./obs_lagged_regression.py ERSST SON 1
./obs_lagged_regression.py ERSST JJA 1
./obs_lagged_regression.py ERSST MAM 1

