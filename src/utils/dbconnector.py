""" QuestDB Connector """

# Base Modules
import io
import json
import requests
import yaml

# 3rd Party Modules
import polars as pl


class Database:
    """
    Klasse zur Verbindung mit QuestDB und zum Abrufen von Daten.

    Attributes:
        _questdb_settings (dict): Einstellungen für die QuestDB-Verbindung.

    """

    def __init__(self, questdb_settings):
        """
        Initialisiert die Datenbank-Verbindungsklasse mit den gegebenen QuestDB-Einstellungen.

        Args:
            questdb_settings (dict): Enthält die Einstellungen für die QuestDB-Verbindung.
        """
        self._questdb_settings = questdb_settings

    def get_data(self, query):
        """
        Ruft Daten anhand einer Abfrage ab.

        Args:
            query (str): Auszuführende Abfrage (muss Spalten für Zeitstempel und Hash enthalten).

        Returns:
            data (polars.DataFrame): Dataframe mit den Daten.
        """
        # Vorbereiten der URL mit DB-Konfiguration und Abfrage
        url = f"http://{self._questdb_settings['host']}:{self._questdb_settings['port']}/exp?query={query}"

        # Daten abrufen
        request = requests.get(url, timeout=600)

        # Lesen der Daten in ein Polars DataFrame
        data = pl.read_csv(io.StringIO(request.text))

        # Umwandlung des Zeitstempels in das datetime-Format
        data = data.with_columns(
            pl.col("timestamp")
            .str.strptime(pl.Datetime, fmt="%Y-%m-%dT%H:%M:%S.%6fZ")
            .alias("timestamp")
        )

        # Ändern von Spalten mit Float64 zu Float32
        for col in data.columns:
            if data[col].dtype == pl.Float64:
                data = data.with_columns(pl.col(col).cast(pl.Float32).alias(col))

        # Sortieren nach Einfüge-Hash und Zeit
        data = data.sort(["hash", "timestamp"])

        # Daten zurückgeben
        return data
