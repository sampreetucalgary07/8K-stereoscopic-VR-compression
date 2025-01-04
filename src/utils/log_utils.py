import csv
import os


def csv_dir_check(file_path, overwrite=False):
    if os.path.exists(file_path):
        if overwrite:
            os.remove(file_path)
            print("\n ----- Previous csv removed ----- \n")
        else:
            raise ValueError(
                "CSV file already exists! Set overwrite=True to overwrite the file"
            )
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print("\n ----- Directory created for csv file ----- \n")


def init_csv(file_path, fieldnames):
    with open(file_path, mode="w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()


def append_to_csv(file_path, data):
    with open(file_path, mode="a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=data.keys())
        writer.writerow(data)
