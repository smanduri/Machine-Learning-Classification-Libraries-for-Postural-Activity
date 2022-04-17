import csv

"""
Class Name: DataSetReader
Functionality: Class Method; Reads the data.csv
Methods: getCsvDataset()
"""


class DataSetReader:
    """
    Method Name: getCsvDataset
    Functionality: Reads the csv files. Returns the Values and Replaces the Tag Identifier
    Return: CSV file format of the Dataset
    """

    @classmethod
    def getCsvDataset(cls):

        tag_identifier = {"010-000-024-033": "ANKLE_LEFT", "010-000-030-096": "ANKLE_RIGHT", "020-000-033-111": "CHEST",
                          "020-000-032-221": "BELT"}
        with open('dataset/data.csv', 'r') as f:
            reader = csv.DictReader(f)
            items = list(reader)

            for item in items:
                item["Tag"] = tag_identifier.get(item["Tag"], None)

            keys = items[0].keys()

            print(keys)

            return items
