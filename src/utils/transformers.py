import pandas as pd
from sklearn.decomposition import PCA


class Transformer_Min_Max:
    """
    Eine Klasse, die eine Min-Max-Transformation auf Daten durchführt.
    """

    def __init__(self, feature_range_min: float, feature_range_max: float) -> None:
        """
        Konstruktor für die Transformer_Min_Max Klasse.

        Args:
            feature_range_min (float): Minimum des Skalierungsbereichs für die Transformation.
            feature_range_max (float): Maximum des Skalierungsbereichs für die Transformation.
        """
        self.feature_range_min = feature_range_min
        self.feature_range_max = feature_range_max
        self.__fit_performed = False

    def fit_transform(self, df: pd.DataFrame):
        """
        Führt eine Min-Max-Transformation auf numerischen Spalten eines DataFrame durch und speichert die notwendigen
        Werte für die Transformationsmethode.

        Args:
            df (pd.DataFrame): DataFrame, das transformiert werden soll.

        Returns:
            pd.DataFrame: DataFrame mit skalierten numerischen Spalten.
        """
        self.numeric_columns = df.select_dtypes(include=["number"]).columns  # Ermittelt die numerischen Spalten.

        numeric_data = df[self.numeric_columns]  # Wählt nur die numerischen Daten aus.

        self.mins = numeric_data.min()  # Berechnet das Minimum für jede numerische Spalte.
        self.maxs = numeric_data.max()  # Berechnet das Maximum für jede numerische Spalte.

        df_copy = df.copy()  # Erstellt eine Kopie des DataFrames, um das Original nicht zu manipulieren.

        # Skaliert die Daten auf den Bereich [feature_range_min, feature_range_max].
        df_copy[numeric_data.columns] = (
            (numeric_data - self.mins) / (self.maxs - self.mins)
        ) * (self.feature_range_max - self.feature_range_min) + self.feature_range_min

        self.__fit_performed = True  # Markiert, dass fit_transform ausgeführt wurde.

        return df_copy

    def transform(self, df: pd.DataFrame):
        """
        Führt eine Min-Max-Transformation auf numerischen Spalten eines DataFrame durch.

        Args:
            df (pd.DataFrame): DataFrame, das transformiert werden soll.

        Returns:
            pd.DataFrame: DataFrame mit skalierten numerischen Spalten.
        """
        if not self.__fit_performed:
            raise Exception("please execute fit_transform first")

        numeric_data = df[self.numeric_columns]  # Wählt nur die numerischen Daten aus.

        df_copy = df.copy()  # Erstellt eine Kopie des DataFrames, um das Original nicht zu manipulieren.

        # Skaliert die Daten auf den Bereich [feature_range_min, feature_range_max].
        df_copy[numeric_data.columns] = (
            (numeric_data - self.mins) / (self.maxs - self.mins)
        ) * (self.feature_range_max - self.feature_range_min) + self.feature_range_min

        return df_copy


class Transformer_Standardize:
    """
    Eine Klasse, die eine Standardisierung auf Daten durchführt.
    """

    def __init__(self) -> None:
        """
        Konstruktor für die Transformer_Standardize Klasse.
        """
        self.__fit_performed = False

    def fit_transform(self, df: pd.DataFrame):
        """
        Führt eine Standardisierung auf numerischen Spalten eines DataFrame durch und speichert die notwendigen
        Werte für die Transformationsmethode.

        Args:
            df (pd.DataFrame): DataFrame, das transformiert werden soll.

        Returns:
            pd.DataFrame: DataFrame mit standardisierten numerischen Spalten.
        """
        self.numeric_columns = df.select_dtypes(include=["number"]).columns  # Ermittelt die numerischen Spalten.

        numeric_data = df[self.numeric_columns]  # Wählt nur die numerischen Daten aus.

        self.means = numeric_data.mean()  # Berechnet den Durchschnitt für jede numerische Spalte.
        self.stds = numeric_data.std()  # Berechnet die Standardabweichung für jede numerische Spalte.

        df_copy = df.copy()  # Erstellt eine Kopie des DataFrames, um das Original nicht zu manipulieren.

        # Standardisiert die Daten.
        df_copy[numeric_data.columns] = (numeric_data - self.means) / self.stds

        self.__fit_performed = True  # Markiert, dass fit_transform ausgeführt wurde.

        return df_copy

    def transform(self, df: pd.DataFrame):
        """
        Führt eine Standardisierung auf numerischen Spalten eines DataFrame durch.

        Args:
            df (pd.DataFrame): DataFrame, das transformiert werden soll.

        Returns:
            pd.DataFrame: DataFrame mit standardisierten numerischen Spalten.
        """
        if not self.__fit_performed:
            raise Exception("please execute fit_transform first")

        numeric_data = df[self.numeric_columns]  # Wählt nur die numerischen Daten aus.

        df_copy = df.copy()  # Erstellt eine Kopie des DataFrames, um das Original nicht zu manipulieren.

        # Standardisiert die Daten.
        df_copy[numeric_data.columns] = (numeric_data - self.means) / self.stds

        return df_copy


class Transformer_PCA:
    """
    Eine Klasse, die eine PCA-Transformation auf Daten durchführt.
    """

    def __init__(self, n_components: int) -> None:
        """
        Konstruktor für die Transformer_PCA Klasse.

        Args:
            n_components (int): Anzahl der Komponenten im Ergebnisraum.
        """
        self.n_components = n_components

    def fit_transform(self, df: pd.DataFrame):
        """
        Führt eine PCA auf numerischen Spalten eines DataFrame durch.

        Args:
            df (pd.DataFrame): DataFrame, das transformiert werden soll.

        Returns:
            pd.DataFrame: DataFrame mit den Komponenten der PCA-Transformation.
        """
        self.numeric_columns = df.select_dtypes(include=["number"]).columns  # Ermittelt die numerischen Spalten.
        self.non_numeric_columns = df.select_dtypes(exclude=["number"]).columns  # Ermittelt die nicht numerischen Spalten.

        self.pca_converter = PCA(n_components=self.n_components)  # Initialisiert den PCA-Konverter.

        # Führt die PCA auf den numerischen Daten durch.
        transformed_data = self.pca_converter.fit_transform(df[self.numeric_columns].to_numpy())

        df_transformed = df[self.non_numeric_columns].copy()

        # Fügt die Komponenten als Spalten mit dem Präfix 'c_' hinzu.
        df_transformed[[f"c_{i}" for i in range(transformed_data.shape[1])]] = transformed_data

        return df_transformed

    def transform(self, df: pd.DataFrame):
        """
        Führt eine PCA auf numerischen Spalten eines DataFrame durch.

        Args:
            df (pd.DataFrame): DataFrame, das transformiert werden soll.

        Returns:
            pd.DataFrame: DataFrame mit den Komponenten der PCA-Transformation.
        """
        # Führt die PCA auf den numerischen Daten durch.
        transformed_data = self.pca_converter.transform(df[self.numeric_columns].to_numpy())

        df_transformed = df[self.non_numeric_columns].copy()

        # Fügt die Komponenten als Spalten mit dem Präfix 'c_' hinzu.
        df_transformed[[f"c_{i}" for i in range(transformed_data.shape[1])]] = transformed_data

        return df_transformed
