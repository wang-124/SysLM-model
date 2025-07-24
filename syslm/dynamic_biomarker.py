#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# dynamic_biomarker.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def identify_dynamic_biomarkers(data, std_threshold='mean', slope_threshold=0.2):
    """
    Identify biomarkers with both high volatility and significant trends over time.

    Args:
        data (pd.DataFrame): Input DataFrame containing columns:
            - Taxonomy
            - Dataset
            - Classification Label
            - tax
            - Time
            - Weight
        std_threshold (str or float): Use 'mean' to filter std above mean, or provide numeric threshold.
        slope_threshold (float): Minimum absolute slope to consider a trend significant.

    Returns:
        volatility_df (pd.DataFrame): Biomarkers with high standard deviation.
        trend_df (pd.DataFrame): Biomarkers with significant trend.
        dynamic_df (pd.DataFrame): Biomarkers satisfying both criteria.
    """

    # === Volatility ===
    grouped = data.groupby(['Taxonomy', 'Dataset', 'Classification Label', 'tax'])
    std_devs = grouped['Weight'].std().fillna(0)

    # Determine threshold
    if std_threshold == 'mean':
        threshold = std_devs.mean()
    else:
        threshold = std_threshold

    volatility_df = std_devs[std_devs > threshold].reset_index()
    volatility_df['Standard Deviation'] = volatility_df['Weight']
    volatility_df = volatility_df.drop(columns=['Weight'])

    # === Trend ===
    trends = []
    for (taxonomy, dataset, label, taxon), group in grouped:
        time_map = {t: i for i, t in enumerate(sorted(group['Time'].unique()))}
        group = group.copy()
        group['Time_num'] = group['Time'].map(time_map)

        X = group['Time_num'].values.reshape(-1, 1)
        y = group['Weight'].values

        if len(X) >= 2:  # At least 2 time points needed
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            trends.append((taxonomy, dataset, label, taxon, slope))

    trend_df = pd.DataFrame(trends, columns=['Taxonomy', 'Dataset', 'Classification Label', 'tax', 'Slope'])
    trend_df = trend_df[trend_df['Slope'].abs() > slope_threshold]

    # === Combine ===
    dynamic_df = pd.merge(
        trend_df,
        volatility_df,
        on=['Taxonomy', 'Dataset', 'Classification Label', 'tax']
    )

    return volatility_df, trend_df, dynamic_df

