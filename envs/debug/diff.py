import os
import random
import re
import zipfile


def compare_and_zip_epochs(path, file_v1, file_v2, zip_name):
    """Compares two files containing epoch points, identifies unique epochs in
    the second file, and zips those files.

    Randomly selects 5 common epochs and compares their corresponding files for
    exact match.
    """

    def read_file(file_path):
        with open(file_path, 'r') as file:
            return set(re.findall(r'epoch\d+', file.read()))

    def zip_files(files_to_zip, zip_file_name):
        with zipfile.ZipFile(zip_file_name, 'w') as zipf:
            for file in files_to_zip:
                zipf.write(file, os.path.basename(file))

    def are_files_identical(file1, file2):
        with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
            return f1.read() == f2.read()

    # Read and extract epoch points from both files
    epochs_v1 = read_file(os.path.join(path, file_v1))
    epochs_v2 = read_file(os.path.join(path, file_v2))

    # Identify unique epochs in v2
    unique_epochs_v2 = epochs_v2 - epochs_v1

    # Prepare file paths for zipping
    files_to_zip = [
        os.path.join(path, epoch + '.mat') for epoch in unique_epochs_v2
        if os.path.exists(os.path.join(path, epoch + '.mat'))
    ]

    # Zip the unique files
    zip_files(files_to_zip, os.path.join(path, zip_name))

    # Randomly select 5 common epochs and compare their corresponding files
    common_epochs = list(epochs_v1 & epochs_v2)
    comparison_results = {}
    for epoch in random.sample(common_epochs, min(5, len(common_epochs))):
        file1 = os.path.join(path, epoch + '.mat')  # Adjusted file paths
        file2 = os.path.join(path, epoch + '.mat')  # Adjusted file paths
        comparison_results[epoch] = are_files_identical(file1, file2)

    return comparison_results


# Example usage

compare_and_zip_epochs('data/M3CV/Testing', 'TestingV1.txt', 'TestingV2.txt',
                       'unique_epochs_v2.zip')
