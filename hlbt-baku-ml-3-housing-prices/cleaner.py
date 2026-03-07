import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np


class HouseCleaner(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.current_year = 2026 + (2 - 1) / 12
        self.preprocessor = None

    def _feature_engineering(self, df):
        df = df.copy()

        df['SaleYearFloat'] = df['YrSold'] + (df['MoSold'] - 1) / 12
        df['YearsSinceSale'] = self.current_year - df['SaleYearFloat']
        df = df.drop(columns=['YrSold', 'MoSold', 'SaleYearFloat'])

        df = df.drop(columns=['PoolQC', 'MiscFeature'])

        return df

    def fit(self, X, y=None):

        X = self._feature_engineering(X)

        quality_order = ['Po','Fa','TA','Gd','Ex']

        ordinal_features = [
            'ExterQual','ExterCond','BsmtQual','BsmtCond',
            'HeatingQC','KitchenQual','GarageQual',
            'GarageCond','FireplaceQu'
        ]

        ordinal_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(
                categories=[quality_order]*len(ordinal_features),
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ])


        nominal_features = [
            'MSZoning','Neighborhood','Condition1','Condition2',
            'BldgType','HouseStyle','RoofStyle','RoofMatl',
            'Exterior1st','Exterior2nd','Foundation',
            'GarageType','SaleType','SaleCondition',
            'LandContour','LotConfig','Heating','Fence'
        ]

        nominal_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])


        numeric_features = X.select_dtypes(exclude="object").columns.tolist()

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])


        self.preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_features),
            ("ord", ordinal_transformer, ordinal_features),
            ("nom", nominal_transformer, nominal_features)
        ])

        self.preprocessor.fit(X)

        return self

    def transform(self, X):

        X = self._feature_engineering(X)

        X_transformed = self.preprocessor.transform(X)

        return X_transformed