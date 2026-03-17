import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class HouseCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.preprocessor = None
        self.ordinal_features_ = None
        self.nominal_features_ = None
        self.numeric_features_ = None

    def _mode_fill(self, series: pd.Series) -> pd.Series:
        mode = series.mode(dropna=True)
        if mode.empty:
            return series
        return series.fillna(mode.iloc[0])

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "MSSubClass" in df.columns:
            df["MSSubClass"] = df["MSSubClass"].astype(str)

        none_fill_map = {
            "Alley": "None",
            "MasVnrType": "None",
            "BsmtQual": "None",
            "BsmtCond": "None",
            "BsmtExposure": "None",
            "BsmtFinType1": "None",
            "BsmtFinType2": "None",
            "FireplaceQu": "None",
            "GarageType": "None",
            "GarageFinish": "None",
            "GarageQual": "None",
            "GarageCond": "None",
            "PoolQC": "None",
            "Fence": "None",
            "MiscFeature": "None",
        }
        for col, fill_value in none_fill_map.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_value)

        zero_fill_cols = [
            "MasVnrArea",
            "BsmtFinSF1",
            "BsmtFinSF2",
            "BsmtUnfSF",
            "TotalBsmtSF",
            "BsmtFullBath",
            "BsmtHalfBath",
            "GarageYrBlt",
            "GarageCars",
            "GarageArea",
        ]
        for col in zero_fill_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
            df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
                lambda s: s.fillna(s.median())
            )
            df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

        mode_fill_cols = [
            "MSZoning",
            "Utilities",
            "Exterior1st",
            "Exterior2nd",
            "Electrical",
            "KitchenQual",
            "Functional",
            "SaleType",
        ]
        for col in mode_fill_cols:
            if col in df.columns:
                df[col] = self._mode_fill(df[col])

        if {"YrSold", "YearBuilt"}.issubset(df.columns):
            df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)

        if {"YrSold", "YearRemodAdd"}.issubset(df.columns):
            df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)

        if {"YearBuilt", "YearRemodAdd"}.issubset(df.columns):
            df["IsRemodeled"] = (df["YearRemodAdd"] > df["YearBuilt"]).astype(int)

        if {"YrSold", "GarageYrBlt"}.issubset(df.columns):
            garage_year = df["GarageYrBlt"].replace(0, np.nan)
            df["GarageAge"] = (df["YrSold"] - garage_year).clip(lower=0).fillna(0)

        if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
            df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

        if {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}.issubset(df.columns):
            df["TotalBath"] = (
                df["FullBath"]
                + 0.5 * df["HalfBath"]
                + df["BsmtFullBath"]
                + 0.5 * df["BsmtHalfBath"]
            )

        porch_cols = [
            col for col in ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
            if col in df.columns
        ]
        if porch_cols:
            df["TotalPorchSF"] = df[porch_cols].sum(axis=1)

        if "Alley" in df.columns:
            df["HasAlley"] = (df["Alley"] != "None").astype(int)
        if "TotalBsmtSF" in df.columns:
            df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
        if "GarageArea" in df.columns:
            df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
        if "Fireplaces" in df.columns:
            df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
        if "PoolArea" in df.columns:
            df["HasPool"] = (df["PoolArea"] > 0).astype(int)
        if "Fence" in df.columns:
            df["HasFence"] = (df["Fence"] != "None").astype(int)
        if "MiscFeature" in df.columns:
            df["HasMiscFeature"] = (df["MiscFeature"] != "None").astype(int)
        if "2ndFlrSF" in df.columns:
            df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)

        return df

    def fit(self, X, y=None):
        X = self._feature_engineering(X)

        ordinal_feature_orders = {
            "LotShape": ["IR3", "IR2", "IR1", "Reg"],
            "LandSlope": ["Sev", "Mod", "Gtl"],
            "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
            "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
            "BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
            "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
            "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
            "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
            "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
            "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
            "FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "GarageFinish": ["None", "Unf", "RFn", "Fin"],
            "GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "PavedDrive": ["N", "P", "Y"],
        }

        self.ordinal_features_ = [
            col for col in ordinal_feature_orders.keys() if col in X.columns
        ]

        categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
        self.nominal_features_ = [
            col for col in categorical_features if col not in self.ordinal_features_
        ]
        self.numeric_features_ = [
            col for col in X.columns if col not in categorical_features
        ]

        ordinal_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[ordinal_feature_orders[col] for col in self.ordinal_features_],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ])

        nominal_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

        transformers = []
        if self.numeric_features_:
            transformers.append(("num", numeric_transformer, self.numeric_features_))
        if self.ordinal_features_:
            transformers.append(("ord", ordinal_transformer, self.ordinal_features_))
        if self.nominal_features_:
            transformers.append(("nom", nominal_transformer, self.nominal_features_))

        self.preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        X = self._feature_engineering(X)
        return self.preprocessor.transform(X)
