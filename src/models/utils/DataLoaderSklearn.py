import pandas as pd
import numpy as np
from dill import load

class DataLoaderSklearn_Tabular:
    """
    Datenmodul für tabellarische Daten mit segment_id als Index.
    Dieses Modul lädt die Daten, teilt sie in Trainings- und Testdaten auf und bereitet sie für die weitere Verarbeitung vor.
    """

    def __init__(self, config, data_filename, train_test_split_filename):
        """
        Initialisiert das DataLoaderSklearn_Tabular-Objekt.

        Args:
            config: Konfigurationsobjekt mit den notwendigen Einstellungen.
            data_filename: Der Pfad zur Datei, die die Daten enthält.
            train_test_split_filename: Der Pfad zur Datei, die die Train-Test-Aufteilung enthält.
        """

        # Liest alle benötigten Parameter
        self.train_test_split_filename = train_test_split_filename
        self.data_filename = data_filename

        # Führt das Setup aus
        self.setup()

    def setup(self):
        """
        Lädt die Daten und teilt sie in Trainings- und Testdaten auf.
        """

        # Lädt die vordefinierte Train-Test-Aufteilung mit pandas
        train_test_split_ids = pd.read_json(
            self.train_test_split_filename, typ="series"
        )

        # Lädt die Datasets
        data = pd.read_parquet(self.data_filename)

        # Teilt die Daten in Trainings- und Testdaten auf und konvertiert sie in numpy Arrays
        temp_train_data = (
            data.loc[train_test_split_ids["train"]]
            .iloc[:, :-3]
            .to_numpy(dtype=np.float32)
        )
        temp_train_labels = np.vectorize(onehotencode.get)(
            data.loc[train_test_split_ids["train"]].iloc[:, -3].to_numpy()
        )
        temp_test_data = (
            data.loc[train_test_split_ids["test"]]
            .iloc[:, :-3]
            .to_numpy(dtype=np.float32)
        )
        temp_test_labels = np.vectorize(onehotencode.get)(
            data.loc[train_test_split_ids["test"]].iloc[:, -3].to_numpy()
        )

        # Speichert die Trainingsdaten
        self.train_data = temp_train_data
        self.train_labels = temp_train_labels

        # Speichert die Testdaten
        self.test_data = temp_test_data
        self.test_labels = temp_test_labels


class DataLoaderSklearn_Segments:
    """
    Datenmodul für segmentierte Daten mit segment_id als Index.
    Dieses Modul lädt die Daten, teilt sie in Trainings- und Testdaten auf und bereitet sie für die weitere Verarbeitung vor.
    """

    def __init__(self, config, data_filename, train_test_split_filename):
        """
        Initialisiert das DataLoaderSklearn_Segments-Objekt.

        Args:
            config: Konfigurationsobjekt mit den notwendigen Einstellungen.
            data_filename: Der Pfad zur Datei, die die Daten enthält.
            train_test_split_filename: Der Pfad zur Datei, die die Train-Test-Aufteilung enthält.
        """

        # Liest alle benötigten Parameter
        self.train_test_split_filename = train_test_split_filename
        self.data_filename = data_filename
        self.config = config.data.get("params")

        # Führt das Setup aus
        self.setup()

    def setup(self):
        """
        Lädt die Daten und teilt sie in Trainings- und Testdaten auf.
        """

        # Lädt die vordefinierte Train-Test-Aufteilung mit pandas
        train_test_split_ids = pd.read_json(
            self.train_test_split_filename, typ="series"
        )

        # Lädt die Datasets
        with open(self.data_filename, "rb") as fr:  # Lädt die Daten
            data: dict = load(fr)

        # Holt die Spalteninformation aus der Konfiguration
        columns = self.config["columns"]

        # Überprüft, ob die Spalten ein String sind
        if type(columns) is str:
            if columns == "numeric":
                # Holt die numerischen Spalten
                columns = (
                    list(data.values())[0][0].select_dtypes(include=["number"]).columns
                )
            else:
                # Wirft einen Fehler, wenn der Spaltenidentifikator unbekannt ist
                raise Exception("unknown column identifier")

        # Holt die Train- und Test-IDs
        train_ids = train_test_split_ids["train"]
        test_ids = train_test_split_ids["test"]

        # Temporäre Speicher für Daten und Labels
        x_train_temp = []
        y_train_temp = []
        x_test_temp = []
        y_test_temp = []

        # Geht durch alle Datenpunkte und teilt sie in Trainings- und Testdaten auf
        for key, segments in data.items():
            activity = key[0]
            for segment in segments:
                segment_id = segment["segment_id"][0]

                # Holt die Merkmale und das Label
                features = segment[columns].to_numpy().transpose().flatten()
                y = onehotencode[activity]

                # Überprüft, ob die segment_id in den Trainings-IDs ist
                if segment_id in train_ids:
                    x_train_temp.append(features)
                    y_train_temp.append(y)

                # Überprüft, ob die segment_id in den Test-IDs ist
                elif segment_id in test_ids:
                    x_test_temp.append(features)
                    y_test_temp.append(y)

                else:
                    # Wirft einen Fehler, wenn die segment_id unbekannt ist
                    raise Exception(
                        f"oh buoy nod gud... unknown segment id: {segment_id}"
                    )

        # Speichert die Trainingsdaten
        self.train_data = x_train_temp
        self.train_labels = y_train_temp

        # Speichert die Testdaten
        self.test_data = x_test_temp
        self.test_labels = y_test_temp


# Lädt die OneHotEncodings
onehotencode = {
    "Sitzen": 0,
    "Laufen": 1,
    "Velofahren": 2,
    "Rennen": 3,
    "Stehen": 4,
    "Treppenlaufen": 5,
}
