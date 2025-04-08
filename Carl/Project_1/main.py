"""Heavily based on Yihuai's code but with some additions/changes."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, cross_validate, KFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def inspect_data(data):
    """Inspecting the data, printing some observations and creating some 
    figures."""

    # Seperate labels from data
    labels = data["V1"].values
    pixels = data.iloc[:, 1:].values
    print("\nSize of entire data set: " + str(round(pixels.shape[0])))

    # Print digit label distribution
    print("\nFrequency distribution:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print("Label " + str(label) + ": " + str(count) + " samples")

    # Define x positions
    x_positions = np.arange(len(unique_labels))
    plt.figure(figsize=(10, 7))
    plt.bar(x_positions, counts)
    plt.xticks(x_positions, unique_labels)
    plt.xlabel("Labels")
    plt.ylabel("Frequency")
    plt.title("Frequency of labels")
    plt.show()

    # Select one random example for each digit label
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    axes = axes.flatten()
    indx = 0
    for lab in unique_labels:
        indices = np.where(labels == lab)[0]
        random_index = np.random.choice(indices)
        pixels_index = pixels[random_index]
        reshaped_data = pixels_index.reshape(16, 16)
        axes[indx].imshow(reshaped_data, cmap='gray', vmin=-1, vmax=1)
        axes[indx].set_title("Digit: " + str(lab))
        axes[indx].axis('off')
        indx = indx + 1
 
    plt.tight_layout()
    plt.suptitle(
        "Random sample of each unique digit in the data set", fontsize=18
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.show()


def solve_task(pixels, labels):
    """Solves task 1/2."""

    # Define classifier in pipeline
    # - A pipeline consists of some pre-processing and a classification model.
    # - Pipeline that contains the 3 models
    #   * (LDA) Linear discriminant analysis. First reduces the 
    #     dimensionality, then finds linear combination of features.
    #   * Logistic regression. 
    #   * (KNN) k neighbours classifier. Assigns label based on the majority 
    #     vote of its nearest neighbors.
    # - The StandarScalar function removes mean and scales to unit variance.
    pipeline = {
        'LDA': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LinearDiscriminantAnalysis())
        ]),
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, solver='lbfgs'))
        ]),
        'kNN': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier())
        ])
    }
    
    # Print classifiers
    print("\nClassifiers:")
    for name, _ in pipeline.items():
        print("- " + name)


     #____Performs_normal_cross_validation_____________________________________
    print("\n________NORMAL_CROSS_VALIDATION______________")
    # Set parameters
    number_of_random_trials = 5
    number_of_k_folds = 10

    # Initialize result dict
    cross_validation_results = {
        'LDA': np.zeros(number_of_random_trials),
        'LogisticRegression': np.zeros(number_of_random_trials),
        'kNN': np.zeros(number_of_random_trials)
    }

    # For each classifier in pipeline
    for name, pipe in pipeline.items():
        print(
            "\n" + name + " classifier using " 
            + str(number_of_k_folds) + "-folds" ":"
        )
        
        # For each random trail
        for i in range(number_of_random_trials):

            # Definie cross validation technique, k-fold 
            k_fold = KFold(
                n_splits=number_of_k_folds, shuffle=True, random_state=i
            )

            # Normal cross validation
            scores = cross_val_score(
                pipe, pixels, labels, cv=k_fold, scoring='accuracy'
            )

            # Store results
            cross_validation_results[name][i] = scores.mean()

            # Print results
            print("- Trial " + str(i + 1) + ": Accuracy "
                + str(round(scores.mean(), 4)) + ", sd " 
                + str(round(scores.std(), 4)))


     #____Performs_double_cross_validation_____________________________________
    print("\n________DOUBLE_CROSS_VALIDATION_WITH_TUNING__")
    
    # Parameter grid
    param_grids = {
        'LDA': {
            'clf__solver': ['svd', 'lsqr', 'eigen']
        },
        'LogisticRegression': {
            'clf__C': [0.01, 0.1, 1, 10, 100]
        },
        'kNN': {
            'clf__n_neighbors': [1, 3, 5, 7, 9] 
        }
    }

    # k-fold parameters
    number_of_outer_k_folds = 10
    number_of_inner_k_folds = 10

    # Initialize result dictionaries
    train_error = {
        'LDA': [None]*number_of_outer_k_folds,
        'LogisticRegression': [None]*number_of_outer_k_folds,
        'kNN': [None]*number_of_outer_k_folds
    }
    test_error = {
        'LDA': [None]*number_of_outer_k_folds,
        'LogisticRegression': [None]*number_of_outer_k_folds,
        'kNN': [None]*number_of_outer_k_folds
    }
    best_params = {
        'LDA': [None]*number_of_outer_k_folds,
        'LogisticRegression': [None]*number_of_outer_k_folds,
        'kNN': [None]*number_of_outer_k_folds
    }

    # Outer k-fold 
    outer_k_fold = KFold(
        n_splits=number_of_outer_k_folds, shuffle=True, random_state=0
    )

    print("\nCalculating...")
    # For each outer k-fold
    for i, (outer_train_index, outer_test_index) in \
        enumerate(outer_k_fold.split(pixels, labels)):
        
        # Divide data into train and test
        pixels_train = pixels[outer_train_index]
        labels_train = labels[outer_train_index]
        pixels_test = pixels[outer_test_index]
        labels_test = labels[outer_test_index]

        # Inner k-fold
        inner_k_fold = KFold(
            n_splits=number_of_inner_k_folds, shuffle=True, random_state=i
        )

        # For each classifier in pipeline
        for name, pipe in pipeline.items():

            # Perform grid search and inner loop
            grid_search = GridSearchCV(
                estimator=pipe,
                param_grid=param_grids[name],
                cv = inner_k_fold,
                scoring='accuracy'
            )
            grid_search.fit(pixels_train, labels_train)
            best_model = grid_search.best_estimator_

            # Training error
            labels_train_predicted = best_model.predict(pixels_train)
            train_error[name][i] = accuracy_score(
                labels_train, labels_train_predicted
            )

            # Test error
            labels_test_predicted = best_model.predict(pixels_test)
            test_error[name][i] = accuracy_score(
                labels_test, labels_test_predicted
            )

            # Store params
            best_params[name][i] = best_model.get_params()

    # Store parameters
    c_list = {}
    c_list["LogisticRegression"] = [
        [
            best_params["LogisticRegression"][i]["clf__C"] \
            for i in range(number_of_outer_k_folds)
        ]
    ]
    c_list["kNN"] = [
        [
            best_params["kNN"][i]["clf__n_neighbors"] 
            for i in range(number_of_outer_k_folds)
        ]
    ]

    # Print some results
    for name, pipe in pipeline.items():
        print(
            "\n" + name + " classifier using " \
            + str(number_of_outer_k_folds) \
            + "-outer folds and " \
            + str(number_of_inner_k_folds) \
            + "-inner folds:"
        )
        print(
            "- Average accuracy " \
            + str(round(np.mean(train_error[name]), 4)) + ", sd " \
            + str(round(np.std(train_error[name]), 4))
        )

        if name in ["LogisticRegression", "kNN"]:
            print(
                "- Average parameter " \
                + str(round(np.mean(c_list[name]), 4)) + ", sd " \
                + str(round(np.std(c_list[name]), 4))
            )

    print("\nPlotting results...")

    # Create box plots of test error for all three models
    model_names = list(test_error.keys())
    errors = [test_error[model] for model in model_names]
    plt.figure(figsize=(10, 7))
    plt.boxplot(errors, tick_labels=model_names)
    plt.ylabel("Test error")
    plt.title("Test error comparison")
    plt.grid(True)
    plt.show()

    # Create box plots of tuning paramers for logistic regression
    plt.figure(figsize=(10, 7))
    plt.boxplot(c_list["LogisticRegression"], tick_labels=["LogisticRegression"])
    plt.ylabel("c")
    plt.title("Logistic regression parameter")
    plt.grid(True)
    plt.show()

    # Create box plots of tuning paramers for kNN
    plt.figure(figsize=(10, 7))
    plt.boxplot(c_list["kNN"], tick_labels=["kNN"])
    plt.ylabel("k")
    plt.title("kNN parameter")
    plt.grid(True)
    plt.show()

    print("Done!")

def task_1(data):
    
    # Seperate labels from data and run model
    labels = data["V1"].values
    pixels = data.iloc[:, 1:].values
    solve_task(pixels, labels)


def task_2(data):
    
    # Seperate labels from data
    labels = data["V1"].values
    pixels = data.iloc[:, 1:].values

    # Run model on each data fraction instance
    data_fraction = [0.5, 0.25, 0.05]
    for frac in data_fraction:
        pixels_train, _, labels_train, _ = train_test_split(
            pixels, labels, test_size=(1-frac), random_state=123, stratify=labels
        )
        print("\nSize of train set: " + str(round(pixels_train.shape[0])))
        solve_task(pixels_train, labels_train)


if __name__ == '__main__':

    data = pd.read_csv('Carl/Project_1/Numbers.txt', sep=r'\s+')
    #inspect_data(data)
    task_1(data)
    #task_2(data)