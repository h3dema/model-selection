# Dataset Subset Generator

This Python project allows you to generate a synthetic dataset `$D$` composed of labeled elements belonging to a set of classes `$C$`. You can then create multiple subsets `$D_i$` from `$D$`, each with a configurable size and without repeating elements. To help analyze the data, the project also visualizes the class distribution within each subset using pie charts.

## Features

- Configurable class set `$C$`: Choose how many distinct classes you want.
- Custom dataset `$D$`: Define the total number of elements.
- Flexible subsets `$D_i$`: Specify different sizes for each subset.
- Random class assignment: Each element in `$D$` is randomly assigned a class from `$C$`.


## How to Use

Check out the [`Sample dataset.ipynb`](./Sample%20dataset.ipynb) notebook for a step-by-step guide on how to use the code.
You can modify the parameters in the notebook to generate datasets and subsets according to your needs.
