import numpy as np
import timesynth as ts

def generate_gp_data(stop_time, variance, lengthscale ,num_points,
                     keep_percentage, noise):
    time_sampler = ts.TimeSampler(stop_time=stop_time)
    irregular_time_samples = time_sampler.sample_irregular_time(num_points=num_points,
                                                                keep_percentage=keep_percentage)
    gp_object = ts.signals.GaussianProcess(variance=variance, lengthscale=lengthscale, kernel='SE')
    white_noise = ts.noise.GaussianNoise(std=noise)
    timeseries = ts.TimeSeries(gp_object, noise_generator=white_noise)
    samples, signals, errors = timeseries.sample(irregular_time_samples)
    # Reshaping to concatenate
    samples = np.reshape(samples, (len(samples), 1))
    irregular_time_samples = np.reshape(irregular_time_samples, (len(irregular_time_samples), 1))
    return np.concatenate((samples, irregular_time_samples), axis=1)

def generate_sin_data(stop_time, amplitude, freq ,num_points, keep_percentage, noise):
    time_sampler = ts.TimeSampler(stop_time=stop_time)
    irregular_time_samples = time_sampler.sample_irregular_time(num_points=num_points,
                                                                keep_percentage=keep_percentage)
    sinusoidal_object = ts.signals.Sinusoidal(frequency=freq, amplitude=amplitude)
    white_noise = ts.noise.GaussianNoise(std=noise)
    timeseries = ts.TimeSeries(sinusoidal_object, noise_generator=white_noise)
    samples, signals, errors = timeseries.sample(irregular_time_samples)
    # Reshaping to concatenate
    samples = np.reshape(samples, (len(samples), 1))
    irregular_time_samples = np.reshape(irregular_time_samples, (len(irregular_time_samples), 1))
    return np.concatenate((samples, irregular_time_samples), axis=1)


def create_batch_dataset(batch_size):
    # Deciding the number of datapoints for each class
    class_dist_array = np.random.choice([0,1], replace=True, size=batch_size)
    # Number of points between 1000 and 2000
    num_points = np.random.randint(1000, 2000, size= batch_size)
    # Sparsity between 25% and 75%
    kp = np.random.choice(np.linspace(25, 75, num=11), size= batch_size)
    # Setting stop time
    stop_time_values = np.random.uniform(low=10., high=50., size = batch_size)
    # Calculating the maximum length
    max_len = np.ceil(np.max(0.01*np.multiply(num_points, kp)))
    # Creating an empty matrix batch_size x max_len x input_dimension
    n_input = 2
    n_output = 2
    X = np.zeros((batch_size, max_len, n_input))
    Y = np.zeros((batch_size, n_output))

    # Sampling parameters for the sinusoidal datapoints
    amplitude_values = np.random.uniform(low=0.5, high=2., size= batch_size)
    freq_values = np.random.uniform(low=0.2, high=2., size= batch_size)
    noise_values = np.random.uniform(low=0, high=0.5, size= batch_size)

    # Sampling parameters for the gaussian datapoints
    lengthscale_values = np.random.uniform(low=0.2, high=1., size= batch_size)
    variance_values = np.random.uniform(low=0.1, high=1., size= batch_size)
    # Populating the master data array
    length_array = np.zeros(batch_size)
    for i in range(batch_size):
        if class_dist_array[i] == 0:
            data = generate_sin_data(stop_time_values[i],
                                     amplitude_values[i],
                                     freq_values[i],
                                     num_points[i],
                                     kp[i], noise_values[i])
            X[i, :len(data), :] = data
            Y[i, 0] = 1
        else:
            data = generate_gp_data(stop_time_values[i],
                                    variance_values[i],
                                    lengthscale_values[i],
                                    num_points[i],
                                    kp[i], noise_values[i])
            X[i, :len(data), :] = data
            Y[i, 1] = 1
        length_array[i] = len(data)

    return X, Y, length_array
