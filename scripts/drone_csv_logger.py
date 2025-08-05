import csv
import os

class DRONECSVLogger:
    def __init__(self, filepath: str, header: list):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, mode="w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=header)
        self.writer.writeheader()

    def log(self, trial, time, drone_id, pkg_id, at_depot, lat, lon, reward, true_return_time,
            true_delivery_time, success_prob, travel_time):
        row = {
            "trial": trial,
            "time": time,
            "drone_id": drone_id,
            "pkg_id": pkg_id,
            "at_depot": at_depot,
            "lat": lat,
            "lon": lon,
            "reward": reward,
            "true_delivery_time": true_delivery_time,
            "true_return_time": true_return_time,
            "success_prob": success_prob,
            "travel_time": travel_time
        }
        self.writer.writerow(row)

    def close(self):
        self.file.close()