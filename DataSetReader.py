import csv


class DataSetReader:

    @classmethod
    def getCsvDataset(cls):
        with open('dataset/data.csv', 'r') as f:
            reader = csv.DictReader(f)
            items = list(reader)
            return items
