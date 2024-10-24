# TP ML

## Contributors

- DEBRAY Clarisse
- FERNANDES DE ALMEIDA Maxsuel
- GLERANT Pierre
- VOGELS Arthur

## Purpose of the project

Predict the quality of welds on steels using a data table that provides information on their chemical composition and mechanical properties.

## Organisation of the repository

data directory : store *.data and *.csv files

src directory : composed of the preprocessing notebook and four other directories (one for each contributors)

## data directory

welddb : initial data
preprocessed_data : preprocessed data obtained from src/preprocessing.ipynb
CD1_target_created : data for Clarisse
data_no_reg_no_dup, data_no_reg, data : data for Arthur
preprocessed_data_2 : A copy of data.csv for Pierre

## src directory

### preprocessing.ipynb

Notebook which generate preprocessed_data.csv

### model_clarisse

XGBoost to predict a binary target (1 if the weld is of good quality, 0 otherwise)

### model_maxuel

Label Propagation to predict the same binary target as Clarisse

### model_pierre

Random Forest to predict the values of the 5 tests (with and without PCA)

### model_arthur

Several models to predict the value of different tests
