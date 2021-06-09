from os import sep
import numpy as np

N_SAMPLES = 100
MEAN = [0, 0]
COV = [(2, 0.4), (0.4, 0.2)]
DELTA = [5, 5]
PATH_TO_DATA_X = "data/raw/data.csv"
PATH_TO_TARGET = "data/raw/target.csv"


def generate_multivariate_normal_values(mean=MEAN, cov=COV,
                                        n_samples=N_SAMPLES,
                                        delta=DELTA):
    mean = np.array(mean)
    delta = np.array(delta)

    x_target_1 = np.random.multivariate_normal(mean, cov, n_samples).T
    x_target_2 = np.random.multivariate_normal(mean + delta, cov, n_samples).T

    full_x = np.concatenate([x_target_1, x_target_2], axis=1)
    y_target = np.array(N_SAMPLES * [0] + N_SAMPLES * [1])
    return full_x.T, y_target


def save_data(x_data, target_y,
              filepath_x_data=PATH_TO_DATA_X,
              filepath_target=PATH_TO_TARGET):
    np.savetxt(filepath_x_data, x_data, delimiter=',')
    np.savetxt(filepath_target, target_y, fmt="%i")


def main():
    x_data, target_y = generate_multivariate_normal_values()
    save_data(x_data, target_y)


if __name__ == "__main__":
    main()
