import os
import scipy.io.wavfile as audio_reader
import numpy as np
from python_speech_features import mfcc
import pandas as pd


def get_features(sub_sample, rate):
    features = mfcc(sub_sample, rate)
    return features


def process_file(filename, classification_class, parent_path="../../data/raw/"):

    print("Processing file " + filename)
    # read in file
    rate, audio_data = audio_reader.read(parent_path +
                                      classification_class + "/" +
                                      filename)
    # discarding first and last 10%
    length = np.shape(audio_data)[0]
    mark = int(0.1 * length)
    audio_data = audio_data[mark:-mark, :]

    # Converting audio to mono and getting features
    audio_data = np.mean(audio_data, axis=1)
    features = get_features(audio_data, rate)

    # Write out to csv file
    pd.DataFrame(features).to_csv("../../data/processed/" + classification_class + ".csv",
                                  mode="a",
                                  header=False,
                                  index=False)

    # Code to batch data and calculate features per batch
    # Not useful as MFCC already is calculated on splits

    # length = np.shape(audio_data)[0]
    # samples = int(length / rate)
    # start = 0
    # end = start + rate * 2
    # for sample_idx in range(1, samples+1):
    #     sub_sample = audio_data[start:end]
    #     features = get_features(sub_sample, rate)
    #     start = end
    #     end += rate * 2
    #     print("Test")


def main():

    raw_parent = "../../data/raw/"
    source_dirs = os.listdir(raw_parent)

    # Processing each directory
    for directory in source_dirs:
        if directory != 'playlists' and directory != 'talk':
            file_list = os.listdir("../../data/raw/" + directory)
            print("Processing samples of class " + directory)
            for filename in file_list:
                # Here each directory is a class, passing it along with the file
                process_file(filename, directory)


if __name__ == "__main__":
    main()
