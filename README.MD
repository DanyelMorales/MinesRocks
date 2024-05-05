# Mines vs. Rocks

## Disclaimer
This repository serves as a homework assignment with the goal of training a model capable of distinguishing between "mines" and "rocks" by analyzing bouncing sonar signals. The majority of the functions and code have been developed by myself for learning purposes. 

Note: It's advised to refrain from using any of this code in production environments due to its instability. The only libraries employed are Numpy, Pandas, and Matplotlib.

Points to denote:

* The primary dataset comprises 208 examples, from which I am selecting 48 examples for testing purposes, representing a subset of the total dataset.
* The Y axis is encoded using a custom one-hot encoding strategy, primarily focusing on encoding rows where a Mine is detected, while all other instances are denoted as 0.
* The Log Loss function is applied as sigmoid during Gradient Descent as activation function.
* For training purposes, I utilize the calculus derivative function of the sigmoid within Gradient Descent to minimize loss.
* Essentially, the approach to detecting mines involves training a binary classification model.

## About dataset
The dataset comprises two files: "sonar.mines" and "sonar.rocks", containing 111 and 97 patterns, respectively. These patterns were acquired by bouncing sonar signals off a metal cylinder and rocks under similar conditions. The sonar signals are frequency-modulated chirps rising in frequency, obtained from various aspect angles spanning 90 degrees for the cylinder and 180 degrees for the rock.

Each pattern consists of 60 numbers ranging from 0.0 to 1.0, representing the energy within specific frequency bands integrated over time. Higher frequencies are integrated over later periods since they are transmitted later during the chirp.

The label associated with each record indicates whether the object is a rock ("R") or a mine (metal cylinder, "M"). The labels are sorted in increasing order of aspect angle but do not directly encode the angle.

 

## Observations

The accuracy of the weights depends on the configured hyperparameters, such as the learning rate and the number of training iterations. However, even with the same hyperparameters, you may notice that the accuracy varies across multiple runs.

Upon loading the original dataset, you'll observe that the mine data are clustered together with the rock rows, mine data rows appears first in the file and rocks at the very end. To address this issue, I perform a shuffle using Pandas at the outset of the Jupyter notebook.

If the training dataset is not well-distributed, it can significantly impact performance and lead to reduced accuracy.

## Reuse Weights

I have already kept a copy of the weights where accuracy reaches at least 70%, it's a regular npy file so you can load it using numpy:
````
w = np.load("cache/rocks_mines_weightsxxxxxxxx.npy")
classifier.test(X_test, Y_test, w)
````
    
## Reuse shuffled dataset
You can utilize the pre-shuffled dataset, where I achieved a 70% accuracy, by loading the CSV file located at "cache/rocks_mines.csv".
