import sys
import joblib
import jpeglib
import pyarrow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.fft import fft, fftshift, fftfreq
from PIL import Image, UnidentifiedImageError
from matrix_prediction import get_luminance_qtable


def is_jpeg(filepath):
    """
    Check the file extension of the input file and throw an error if it is not JPEG.
    """
    try:
        img = Image.open(filepath)
        if img.format != "JPEG":
            raise IOError("Error: The file is not in JPEG format.")
    except UnidentifiedImageError:
        raise IOError("Error: The file is not a valid image or cannot be opened.")

def extract_dct_blocks(path):
    """
    Read the DCT blocks from the given input image and return the frequency values.
    """
    res_dct = jpeglib.read_dct(path)

    # List with 64 elements for each DCT index
    values = [[] for _ in range(64)]

    for bi in range(res_dct.Y.shape[0]):
        for bj in range(res_dct.Y.shape[1]):
            for i in range(res_dct.Y.shape[2]):
                for j in range(res_dct.Y.shape[3]):
                    index = i * 8 + j
                    values[index].append(res_dct.Y[bi, bj, i, j])

    return np.array(values)

def create_dct8x8(inp_values):
    """
    Create the 8x8 blocks from the frequency values.
    """
    dct8x8 = []

    for idx in range(64):
        data = np.array(inp_values[idx])
        dct8x8.append(data)

    return dct8x8

def filter_ac_components(inp_dct8x8, inp_ac_components_indexes):
    """
    Keep only the frequencies (AC components) at the indices provided in the input.
    """
    ac_components = []

    for i in range(len(inp_ac_components_indexes)):
        ac_components.append(inp_dct8x8[inp_ac_components_indexes[i]])

    return ac_components

def create_dct_histograms(inp_hists):
    """
    Generating histograms for the DCT values.
    """
    histograms = []
    edges = []

    for data in inp_hists:
        hist, bin_edges = np.histogram(data, bins=BINS)
        histograms.append(hist)
        edges.append(bin_edges)

    return histograms, edges

def show_dct_histograms(inp_hists, inp_array_of_bin_edges, title):
    """
    Displaying the histogram with the given title.
    """
    for i, hist in enumerate(inp_hists):
        bin_edges = inp_array_of_bin_edges[i]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.bar(bin_centers, hist, width=np.diff(bin_edges), edgecolor='black', alpha=0.7)

        plt.xlabel('Values')
        plt.ylabel('Density')
        plt.title(f'{title}: {i+1}')
        plt.grid(True)
        plt.show()

def smoothing(inp_hists, inp_filter_size):
    """
    Smoothing the data with the filter size provided in the input.
    """
    smoothed_dct8x8 = []
    kernel = np.ones(inp_filter_size) / inp_filter_size

    for data in inp_hists:
        smoothed_data = np.convolve(data, kernel, mode='same')
        smoothed_dct8x8.append(smoothed_data)

    return smoothed_dct8x8

def subtract_min_of_neighbors_in_n_range(inp_hists, n):
    """
    Noise-reducing averaging filter by subtracting the minimum value of the neighboring frequencies,
    applied only in the direction of the DC frequency.
    N denotes the length of the minimum filter.
    """
    result = []

    for row in inp_hists:
        processed_row = np.zeros(row.shape)

        for i in range(len(row)):
            # Determine the reverse-looking range
            start_index = max(0, i - n + 1)
            window = row[start_index:i + 1]

            # Minimum value in the window
            min_value = np.min(window)

            # Calculating the result: |H_i(f) - M_i(f)|
            processed_value = abs(row[i] - min_value)
            processed_row[i] = processed_value

        result.append(processed_row)

    return np.array(result)

def apply_hanning_window(inp_hists):
    """
    Applying a Hanning window to remove the margin parts.
    """
    hanning_windowed = []

    for data in inp_hists:
        hann = np.hanning(len(data))
        windowed_data = data * hann
        hanning_windowed.append(windowed_data)

    return hanning_windowed

def create_ffts(inp_hists):
    """
    Calculating the FFT (Fast Fourier Transform) values and frequencies."
    """
    fft_results = []
    freq_results = []

    for i, data in enumerate(inp_hists):
        zero_mean_data = data - np.mean(data)

        # Calculating the FFT
        fft_result = np.abs(fft(zero_mean_data))

        # Shifting the FFT
        fft_shifted = fftshift(fft_result)

        # Calculating the frequency axis.
        data_length = len(data)
        freqs = fftfreq(data_length, d=1)
        freqs_shifted = fftshift(freqs)

        fft_results.append(fft_shifted)
        freq_results.append(freqs_shifted)

    return fft_results, freq_results

def normalize(inp_array):
    """
    Normalization of the data (dividing by the maximum).
    """
    return [hist / np.max(hist) for hist in inp_array]

def show_fft_plots(fft_result, freq_results):
    """
    Displaying the FFT values on a diagrams.
    """
    for i, (data, freqs) in enumerate(zip(fft_result, freq_results)):
        plt.figure(figsize=(10, 6))

        plt.plot(freqs, data)

        plt.title(f'FFT of {i + 1} component')
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')

        plt.show()

def create_feature_vectors_multiple_acs(input_array, double_or_not, dataframe, ac_components_number, window_size):
    """
    Creating feature vectors of size AC_COMPONENST_NUMBER (9) * BINS (128) and appending them to the specified DataFrame.
    The feature vectors contain the values of the peak positions within the window_size
    sized window at the corresponding positions.
    The flag variable can be set to indicate whether the image is doubly (True) or singly (False) compressed.
    """
    half_window = window_size // 2

    first_element = 1 if double_or_not else 0
    new_row = [first_element]

    def extract_features(row):
        features = [0] * len(row)
        for i in range(half_window, len(row) - half_window):
            window = row[i - half_window:i + half_window + 1]
            center_value = row[i]
            if center_value == max(window) and list(window).count(center_value) == 1:
                features[i] = center_value
        return features

    for i in range(0, ac_components_number):
        element = extract_features(input_array[i])
        new_row += element

    dataframe.loc[len(dataframe)] = new_row
    return dataframe


BINS = 128
AC_COMPONENTS_INDEXES = [1, 5, 6, 2, 4, 7, 3, 8, 9]
AC_COMPONENTS_NUMBER = 9

def complete_the_process(input_path, df, ac_components_indexes, double_or_not,
                         ac_components_number=9, n=7, filter_size=11, window_size=5):
    # Extracting DCT values
    values = extract_dct_blocks(input_path)

    # Creating the DCT8x8 structure
    dct8x8 = create_dct8x8(values)

    # Filtering only the AC components
    dct8x8 = filter_ac_components(dct8x8, ac_components_indexes)

    hists, array_of_bin_edges = create_dct_histograms(dct8x8)
    #show_dct_histograms(normalize(hists), array_of_bin_edges, "Simple")

    # Averaging filter
    hists = smoothing(hists, filter_size)
    #show_dct_histograms(normalize(hists), array_of_bin_edges, "Smoothed")

    # Subtracting the minimum
    hists = subtract_min_of_neighbors_in_n_range(hists, n)
    #show_dct_histograms(normalize(hists), array_of_bin_edges, "Minimum subtraction")

    # Trimming the margin parts
    hists = apply_hanning_window(hists)
    #show_dct_histograms(normalize(hists), array_of_bin_edges, "Windowed")

    # Calculating and displaying the FFTs
    ffts, freqs = create_ffts(hists)
    #show_fft_plots(normalize(hists), freqs)
    ffts = normalize(ffts)

    create_feature_vectors_multiple_acs(ffts, double_or_not, df, ac_components_number, window_size)


def main():
    # Reading an image from an environment variable.
    input_jpeg_path = sys.argv[1]

    # Determining the image format.
    try:
        is_jpeg(input_jpeg_path)
    except IOError as err:
        print(err)

    # Determination of Quantization Factor.
    matrix_prediction_model = joblib.load('matrix.joblib')
    luminance_table = [get_luminance_qtable(input_jpeg_path)]

    q_table_columns = [f'v{i}' for i in range(64)]
    q_table_df = pd.DataFrame(luminance_table, columns=q_table_columns)

    quantization_factor_prediction = matrix_prediction_model.predict(q_table_df)[0]

    # Model loading.
    jpeg_detection_model = joblib.load(f'models/2_2/q{quantization_factor_prediction}_model_2_2.joblib')

    # Image Computation and Feature Vector Determination.
    columns = ['flag'] + [f'v{i}' for i in range(AC_COMPONENTS_NUMBER * BINS)]
    image_df = pd.DataFrame(columns=columns)
    complete_the_process(input_jpeg_path, df=image_df, ac_components_indexes=AC_COMPONENTS_INDEXES, double_or_not=None)

    # Prediction and result output.
    x_value = image_df.drop(columns=['flag'])
    result = jpeg_detection_model.predict(x_value)
    print(f'The input image is {"doubly" if result else "singly"} compressed.')

if __name__ == "__main__":
    main()
