import pandas as pd
from sklearn.decomposition import PCA


class Transformer_Min_Max:
    """Executes a min_max transformation"""

    def __init__(self, feature_range_min: float, feature_range_max: float) -> None:
        """This class performs a min max transformation and scales to a new range
        Args:
            feature_range_min (float): data minimum scaled to this value
            feature_range_max (float): data maximum scaled to this value
        """

        self.feature_range_min = feature_range_min
        self.feature_range_max = feature_range_max
        self.__fit_performed = False

    def fit_transform(self, df: pd.DataFrame):
        """Execute transformation (only on numeric columns) and store values required for transform
        Args:
            df (pd.DataFrame): data to transform

        Returns:
            transformed DataFrame
        """

        self.numeric_columns = df.select_dtypes(
            include=["number"]
        ).columns  # get numeric columns

        numeric_data = df[self.numeric_columns]  # only get data from numeric columns

        # get mins
        self.mins = numeric_data.min()
        # get maxs
        self.maxs = numeric_data.max()

        df_copy = df.copy()  # copy dataframe to not manipulate the original one

        # min-max scale
        df_copy[numeric_data.columns] = (numeric_data - self.mins) / (
            self.maxs - self.mins
        )
        # scale to feature range
        df_copy[numeric_data.columns] = (
            df_copy[numeric_data.columns]
            * (self.feature_range_max - self.feature_range_min)
            + self.feature_range_min
        )

        self.__fit_performed = True  # set flag

        return df_copy  # return dataframe with scaled numeric columns

    def transform(self, df: pd.DataFrame):
        """Execute transformation (only on numeric columns)
        Args:
            df (pd.DataFrame): data to transform

        Returns:
            transformed DataFrame
        """

        if not self.__fit_performed:
            raise Exception("please execute fit_transform first")

        numeric_data = df[self.numeric_columns]  # only get data from numeric columns

        df_copy = df.copy()  # copy dataframe to not manipulate the original one

        # min-max scale
        df_copy[numeric_data.columns] = (numeric_data - self.mins) / (
            self.maxs - self.mins
        )
        # scale to feature range
        df_copy[numeric_data.columns] = (
            df_copy[numeric_data.columns]
            * (self.feature_range_max - self.feature_range_min)
            + self.feature_range_min
        )

        return df_copy  # return dataframe with scaled numeric columns


class Transformer_Standardize:
    """Executes a standardization"""

    def __init__(self) -> None:
        """This class performs a standardization"""

        self.__fit_performed = False

    def fit_transform(self, df: pd.DataFrame):
        """Execute standardization (only on numeric columns) and store values required for transform
        Args:
            df (pd.DataFrame): data to transform

        Returns:
            transformed DataFrame
        """

        self.numeric_columns = df.select_dtypes(
            include=["number"]
        ).columns  # get numeric columns

        numeric_data = df[self.numeric_columns]  # only get data from numeric columns

        # get means
        self.means = numeric_data.mean()
        # get stds
        self.stds = numeric_data.std()

        df_copy = df.copy()  # copy dataframe to not manipulate the original one

        # standardize
        df_copy[numeric_data.columns] = (numeric_data - self.means) / self.stds

        self.__fit_performed = True  # set flag

        return df_copy  # return dataframe with scaled numeric columns

    def transform(self, df: pd.DataFrame):
        """Execute transformation (only on numeric columns)
        Args:
            df (pd.DataFrame): data to transform

        Returns:
            transformed DataFrame
        """

        if not self.__fit_performed:
            raise Exception("please execute fit_transform first")

        numeric_data = df[self.numeric_columns]  # only get data from numeric columns

        df_copy = df.copy()  # copy dataframe to not manipulate the original one

        # standardize
        df_copy[numeric_data.columns] = (numeric_data - self.means) / self.stds

        return df_copy  # return dataframe with scaled numeric columns


class Transformer_PCA:
    """Send observations to eigenspace"""

    def __init__(self, n_components: int) -> None:
        """This class performs a PCA on numeric observations
        Args:
            n_components (int): number of components in subspace
        """
        self.n_components = n_components

    def fit_transform(self, df: pd.DataFrame):
        """Execute PCA on numeric columns of df
        Args:
            df (pd.DataFrame): data to transform

        Returns:
            transformed DataFrame
        """

        self.numeric_columns = df.select_dtypes(
            include=["number"]
        ).columns  # get numeric columns
        self.non_numeric_columns = df.select_dtypes(
            exclude=["number"]
        ).columns  # get non numeric columns

        self.pca_converter = PCA(
            n_components=self.n_components
        )  # init converter with n_components

        transformed_data = self.pca_converter.fit_transform(
            df[self.numeric_columns].to_numpy()
        )  # fit and transform on numeric data

        df_transformed = df[self.non_numeric_columns].copy()

        df_transformed[
            [f"c_{i}" for i in range(transformed_data.shape[1])]
        ] = transformed_data  # append components as columns with prefix 'c_'

        return df_transformed

    def transform(self, df: pd.DataFrame):
        """Execute transformation (only on numeric columns)
        Args:
            df (pd.DataFrame): data to transform

        Returns:
            transformed DataFrame
        """

        transformed_data = self.pca_converter.transform(
            df[self.numeric_columns].to_numpy()
        )  # transform on numeric data

        df_transformed = df[self.non_numeric_columns].copy()

        df_transformed[
            [f"c_{i}" for i in range(transformed_data.shape[1])]
        ] = transformed_data  # append components as columns with prefix 'c_'

        return df_transformed
