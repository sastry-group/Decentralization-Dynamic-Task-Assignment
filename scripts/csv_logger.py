import csv
import os

class CSVLogger:
    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True) 
        self.files = {}

    def log(self, filename, row, header=None):
        path = os.path.join(self.filepath, filename)
        file_exists = os.path.isfile(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header or row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def close(self):
        pass