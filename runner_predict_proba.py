import os
import pickle
import time

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError(
                f'DataFrame does not contain the following columns: {cols_error}')


class AddFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features, silent=True):
        self.features = features
        self.silent = silent

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.silent:
            start_t = time.time()
            print('Start adding features'.center(100, '*'))
        assert isinstance(X, pd.DataFrame), 'This is not a pandas dataframe'

        X_features = self.features.loc[self.features.index.isin(
            X.index.unique())]

        X_features = X_features.sort_values('buy_time') \
            .groupby('id').last()

        X_merge = X.reset_index() \
            .merge(X_features.reset_index(),  on=X.index.name, how='left', suffixes=('_train', '_features')) \
            .set_index(X.index.name)

        assert X_merge.shape[0] == X.shape[
            0], f'Shapes of dataframe don\'t match: {X_merge.shape[0]} and {X.shape[0]}'
        assert (X_merge.index == X.index).all(), 'Index Sort Error'
        if not self.silent:
            print(
                f'End adding features, run time: {time_format(time.time()-start_t)}'.center(100, '*'))
            print()

        return X_merge


class MemUseOptimizing(BaseEstimator, TransformerMixin):
    def __init__(self, silent=True):
        self.silent = silent

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        start_t = time.time()

        assert isinstance(X, pd.DataFrame), 'This is not a pandas dataframe'

        if not self.silent:
            print('Start of dataframe memory use optimizing'.center(100, '*'))
            start_memory_usage = X.memory_usage(deep=True).sum() / 1024**2

            X_dtype = pd.DataFrame(
                X.dtypes, columns=['dtype'], index=X.columns)

            X_dtype['min'] = X.select_dtypes(['int', 'float']).min()
            X_dtype['max'] = X.select_dtypes(['int', 'float']).max()
            X_dtype['is_int'] = ~(X.select_dtypes(['int', 'float']).astype(
                int).sum() - X.select_dtypes(['int', 'float']).sum()).astype('bool_')

            X_dtype.loc[(X_dtype['is_int'] == True), 'dtype'] = 'int64'
            X_dtype.loc[(X_dtype['is_int'] == True) & (X_dtype['min'] >= np.iinfo(
                'int32').min) & (X_dtype['max'] <= np.iinfo('int32').max), 'dtype'] = 'int32'
            X_dtype.loc[(X_dtype['is_int'] == True) & (X_dtype['min'] >= np.iinfo(
                'int16').min) & (X_dtype['max'] <= np.iinfo('int16').max), 'dtype'] = 'int16'
            X_dtype.loc[(X_dtype['is_int'] == True) & (X_dtype['min'] >= np.iinfo(
                'int8').min) & (X_dtype['max'] <= np.iinfo('int8').max), 'dtype'] = 'int8'

            X_dtype.loc[(X_dtype['is_int'] == True) & (
                X_dtype['min'] >= np.iinfo('uint64').min), 'dtype'] = 'uint64'
            X_dtype.loc[(X_dtype['is_int'] == True) & (X_dtype['min'] >= np.iinfo(
                'uint32').min) & (X_dtype['max'] <= np.iinfo('uint32').max), 'dtype'] = 'uint32'
            X_dtype.loc[(X_dtype['is_int'] == True) & (X_dtype['min'] >= np.iinfo(
                'uint16').min) & (X_dtype['max'] <= np.iinfo('uint16').max), 'dtype'] = 'uint16'
            X_dtype.loc[(X_dtype['is_int'] == True) & (X_dtype['min'] >= np.iinfo(
                'uint8').min) & (X_dtype['max'] <= np.iinfo('uint8').max), 'dtype'] = 'uint8'

            X_dtype.loc[(X_dtype['is_int'] == True) & (X_dtype['min'] == 0) & (
                X_dtype['max'] == 1), 'dtype'] = 'bool_'

            X_dtype.loc[(X_dtype['is_int'] == False), 'dtype'] = 'float64'
            X_dtype.loc[(X_dtype['is_int'] == False) & (X_dtype['min'] >= np.finfo(
                'float32').min) & (X_dtype['max'] <= np.finfo('float32').max), 'dtype'] = 'float32'
            X_dtype.loc[(X_dtype['is_int'] == False) & (X_dtype['min'] >= np.finfo(
                'float16').min) & (X_dtype['max'] <= np.finfo('float16').max), 'dtype'] = 'float16'

            for col in X.select_dtypes('object').columns:
                num_unique_values = len(X[col].unique())
                num_total_values = len(X[col])
                if num_unique_values / num_total_values < 0.5:
                    X_dtype.loc[col, 'dtype'] = 'category'

            dtype = X_dtype['dtype'].to_dict()

            X = X.astype(dtype)

        if not self.silent:
            memory_usage = X.memory_usage(deep=True).sum() / 1024**2
            print('Memory use optimizing'.center(100, '*'))
            print(
                f'Memory usage of properties dataframe before optimizing: {start_memory_usage:.02f} MB')
            print(
                f'Memory usage of properties dataframe after optimizing: {memory_usage:.02f} MB')
            print(
                f'This is {100*memory_usage/start_memory_usage:.02f} % of the initial size')
            print(
                f'End of dataframe memory use optimizing, run time: {time_format(time.time()-start_t)}'.center(64, '*'))
            print()

        return X


class GetDate(BaseEstimator, TransformerMixin):
    def __init__(self, silent=True):
        self.silent = silent

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.silent:
            start_t = time.time()
            print('Start geting date from timestamp'.center(100, '*'))
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)

        assert isinstance(
            X, pd.DataFrame), 'This is not a pandas dataframe or series'

        df = pd.DataFrame()

        for col in X.columns:
            df[f'{col}_day'] = pd.to_datetime(X[col], unit='s').dt.day
            df[f'{col}_month'] = pd.to_datetime(X[col], unit='s').dt.month
            df[f'{col}_week'] = pd.to_datetime(X[col], unit='s').dt.week

        if not self.silent:
            print(
                f'End geting date from timestamp, run time: {time_format(time.time()-start_t)}'.center(100, '*'))
            print()
        return df


TARGET = 'target'

df = pd.read_csv('data_test.csv', index_col=[1]) \
    .drop('Unnamed: 0', axis=1)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

df[TARGET] = model.predict_proba(df)[:, 1]

df.to_csv('answers_test.csv')
