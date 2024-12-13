#!/usr/bin/env python
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Выполняет полную обработку данных, включая:
    - Удаление нулевых строк, дубликатов и пропусков
    - Добавление новых признаков
    - Применение логарифмирования
    - Масштабирование признаков

    Parameters:
        df (pd.DataFrame): Исходный DataFrame для обработки.

    Returns:
        pd.DataFrame: Обработанный DataFrame.
    """


    def duplicates_nan(df):
        zero_rows = df[(df == 0).all(axis=1)]
        print(f'Количество нулевых строк: {zero_rows.shape[0]}')

        df = df[~(df == 0).all(axis=1)]  # Удаляем строки, где все значения нули

        if df.isnull().values.sum() > 0:
            df = df.dropna(how='any')  # Удаляем строки с пропущенными значениями

        if df.duplicated().sum() > 0:
            df = df.drop_duplicates(keep='first')  # Удаляем дубликаты

        return df


    def add_features(df):
        df['BMI'] = df['weight(kg)'] / ((df['height(cm)'] / 100) ** 2)
        df['waist_height_ratio'] = df['waist(cm)'] / df['height(cm)']
        df['chol_ratio'] = df['LDL'] / df['HDL']
        return df


    def log_func(df):
        columns_log = ['Cholesterol', 'triglyceride', 'LDL', 'HDL', 'AST', 'ALT', 'Gtp']
        df[columns_log] = df[columns_log].map(lambda x: x if x > 0 else 1)
        df[columns_log] = np.log(df[columns_log])
        return df


    def scale_features(df):
        columns_scale = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)',
                         'eyesight(right)', 'systolic', 'relaxation', 'fasting blood sugar',
                         'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin',
                         'serum creatinine', 'AST', 'ALT', 'Gtp', 'chol_ratio', 'BMI']
        scaler = StandardScaler()
        df[columns_scale] = scaler.fit_transform(df[columns_scale])
        return df

    df = duplicates_nan(df)  # Удаление нулевых строк, пропусков и дубликатов
    df = add_features(df)    # Добавление новых признаков
    df = log_func(df)        # Логарифмирование
    df = scale_features(df)  # Масштабирование

    return df




