from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

from dataset.dataset import create_classes, create_dataset, create_subsets, assign_classes
from models.models import generate_models
from energy.energy import solar_day_curve_with_weather


if __name__ == "__main__":

    # parameters

    # --- Dataset parameters ---
    N = 1000  # number of samples
    num_classes = 5  # number of classes
    subset_sizes = [100]  # size of subsets

    # --- Model parameters ---
    M = 10  # number of models

    # --- Location parameters for solar generation ---
    latitude, tilt, azimuth = 50, 30, 0   # example

    start_day = 200  # e.g., day of year
    area = 0.248213  # from Panasonic AM-5608CAR solar panel
    eta_ref=0.0005  # 0.05% efficiency for testing low power scenarios

    # --- Other parameters ---
    delta = 1  # 1 second between time steps
    T = 10000  # number of time steps

    # create classes
    classes = create_classes(num_classes)
    print("Classes:", classes)

    # create dataset
    dataset = create_dataset(N)
    print("Dataset size:", len(dataset))

    # assign classes to dataset
    labeled_data = assign_classes(dataset, classes)
    print("Labeled data size:", len(labeled_data))

    # create subsets
    subsets = create_subsets(dataset, subset_sizes)
    print("Number of subsets:", len(subsets))

    # create models
    models = generate_models(M=M, C=classes)
    print(f"Generated {len(models)} models.")

    # generate weather
    # TODO: set start hour, compute number of days based on start hour and T, accumulate results
    hours = np.arange(0, 24, 1)  # hourly intervals
    for d in range(7):
        n = start_day + d
        curve, state = solar_day_curve_with_weather(n, hours, latitude, tilt, azimuth, area=area, eta_ref=eta_ref)
