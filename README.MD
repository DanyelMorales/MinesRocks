# Mines vs. Rocks
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