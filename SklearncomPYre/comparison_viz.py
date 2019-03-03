import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# coding: utf-8



def comparison_viz(comparison, choice = "accuracy"):
    '''
    The purpose of the function is to help User visualize the comparison of accuracies or time given in the comparison
    dataframe. It takes in a dataframe with 7 attributes i.e. model name, training & test scores, model variance,
    and the time it takes to fit, predict model and total time taken for both fit and predict.

    The function outputs(saves in the root directory) a beautiful <a href="https://matplotlib.org">matplotlib</a>
    bar chart comparison of different models' training and test scores or the time it takes to fit and predict.

    Parameters :

    Inputs:

    comparison : Dataframe with 7 columns
    - regressor or classifier name
    - training accuracy
    - test accuracy
    - model variance
    - time it takes to fit
    - time it takes to predict and
    - total time. Type: `pandas.Dataframe
    - Choice of `accuracy` or `time`. Defaults to `accuracy`. Type: `string`

    Outputs:
    - Bar chart of accuracies or time comparison by models. Type: `png`

	inspiration: https://matplotlib.org/gallery/statistics/barchart_demo.html
    '''

    ## Tests

    # Choice Type

    if type(choice) != str:
        raise TypeError("Choice must be of type string")

	# Choice value

    if (choice != 'time') and (choice != 'accuracy'):
        raise ValueError("Choice must either be 'time' or 'accuracy'")

	# Comparison Type

    if type(comparison) != pd.core.frame.DataFrame:
        raise TypeError("Comparison must be of type pandas.core.frame.DataFrame")

	# Comparison Value Rows

    if comparison.shape[0] < 1:
        raise ValueError("Comparison must at least have 1 row")

	# Comparison Value Columns

    if comparison.shape[1]!= 7:
        raise ValueError("Comparison must contain 7 columns (excluding index)")

    # Comparison Models Column Type

    for j in np.arange(0,comparison.shape[0] -1,1):
        if type(comparison.iloc[:,0][j]) != type("string"):
           raise TypeError("Comparison Models column must only contain type string")


    # Comparison Remainder Column Type

    if (comparison.iloc[:,2:8].dtypes == float).all() != True: #pragma: no cover
        raise TypeError("Comparison columns excluding 'Models' must only contain type float")

	## Function
    n_models = comparison.shape[0]

    if choice == 'accuracy': #pragma: no cover
        x = comparison.iloc[:,1]
        y = comparison.iloc[:,2]
        labels = ('Train Accuracy','Test Accuracy','Accuracy','Train and Test Accuracy by Model')
    elif choice == 'time': #pragma: no cover
        x = comparison.iloc[:,4]
        y = comparison.iloc[:,5]
        labels = ('Fit Time','Predict Time','Time (s)','Fit and Predict Time by Model')

    fig, ax = plt.subplots()

    index = np.arange(n_models)
    bar_width = 0.35
    opacity = 0.4

    rects1 = ax.bar(index, x,bar_width,
                    alpha=opacity, color='b',
                    label=labels[0])

    rects2 = ax.bar(index + bar_width, y, bar_width,
                    alpha=opacity, color='r',
                    label=labels[1])

    ax.set_xticklabels(comparison.iloc[:,0])
    ax.set_xlabel('Models')
    ax.set_ylabel(labels[2])
    ax.set_xticks(index + bar_width / 2)
    ax.set_title(labels[3])
    ax.legend()

    fig.tight_layout()

    plt.savefig('comparison.png', bbox_inches='tight')
