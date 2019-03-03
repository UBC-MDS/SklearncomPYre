import os
import sys
import pandas as pd
import pytest
import glob
import matplotlib as plt

sys.path.insert(0, os.path.abspath("../SklearncomPYre"))

from SklearncomPYre import comparison_viz

# Test Dataframe
df = pd.DataFrame({'index': [0, 1, 2, 3],
                   'models': ["Random Forest","MLPClassifier", "Logistic Regression","GaussianNB"],
                   'Train Accuracy': [90.1, 82.2, 73.3, 74.4],
                   'Test Accuracy': [85.4, 78.3, 69.2, 70.1],
                   'Variance': [5.1, 3.4, 5.6, 3.2],
                   'Fit Time (s)': [3, 2, 2.5, 1.2],
                   'Predict Time (s)': [1.1, 1.4, 0.9, 2.2],
                   'Total Time (s)': [4.5, 6.6, 4.3, 2.1]
                  })

df2 = pd.DataFrame({'0_Models': ["Random Forest","MLPClassifier", "Logistic Regression","GaussianNB"],
                   'Train Accuracy': [90.1, 82.2, 73.3, 74.4],
                   'Test Accuracy': [85.4, 78.3, 69.2, 70.1],
                   'Variance': [5.1, 3.4, 5.6, 3.2],
                   'Fit Time (s)': [3, 2, 2.5, 1.2],
                   'Predict Time (s)': [1.1, 1.4, 0.9, 2.2],
                   'Total Time (s)': [4.5, 6.6, 4.3, 2.1]
                  })

def test_input_choice_value():
    """
    Testing for entering an invalid string as a choice.
    """
    try:
        comparison_viz(comparison=df, choice = 'model')
    except(ValueError):
        assert True
    else: #pragma: no cover
        assert False

def test_input_choice_type():
    """
    Testing for entering an invalid type for choice.
    """
    try:
        comparison_viz(comparison=df, choice=3)
    except(TypeError):
        assert True
    else: #pragma: no cover
        assert False


def test_input_comparison_value():
    """
    Testing for entering an invalid dataframe as a comparison.
    """
    try:
        comparison_viz(comparison=df[['models','Variance']], choice='time')
    except(ValueError):
        assert True
    else: #pragma: no cover
        assert False

def test_input_comparison_type():
    """
    Testing for entering an invalid type for choice.
    """
    try:
        comparison_viz(comparison=3, choice='time')
    except(TypeError):
        assert True
    else: #pragma: no cover
        assert False

def test_output_comparison_viz(tmpdir):
    """
    Testing for output from appropriate comprison dataframe and choice.
    """
    comparison_viz(comparison=df2, choice='time')
    pathway = tmpdir.join('comparison.png')
    file = glob.glob(pathway.strpath)
    assert type(file) == list

def test_input_df_dim():
    """
    Testing that given df has 7 columns
    """
    with pytest.raises(ValueError):
        comparison_viz(df, 'time')

def test_input_df_without_rows():
    """
    Comparison dataframe has no rows
    """
    try:
        comparison_viz(comparison=df2.drop([0,1,2,3]), choice='time')
    except(ValueError):
        assert True
    else: #pragma: no cover
        assert False

def test_input_df_model_column_not_string():
    """
    Comparison dataframe model column not string
    """
    try:
        df2['0_Models'] = 1
        comparison_viz(comparison=df2, choice='time')
    except(TypeError):
        assert True
    else: #pragma: no cover
        assert False

def test_input_df_other_columns_not_float():
    """
    Comparison dataframe other columns not float
    """
    try:
        df2['Variance'] = "a"
        comparison_viz(comparison=df2, choice='time')
    except(TypeError):
        assert True
    else: #pragma: no cover
        assert False
