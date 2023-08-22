import numpy as np
import pandas as pd

dataset = pd.read_excel('resources/preprocessed/dataset_completo.xlsx')

def detect_outliers_iqr(data):
    outliers = []
    data = sorted(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    # print(q1, q3)
    IQR = q3-q1
    lwr_bound = q1-(1.5*IQR)
    upr_bound = q3+(1.5*IQR)
    # print(lwr_bound, upr_bound)
    for i in data: 
        if (i<lwr_bound or i>upr_bound):
            outliers.append(i)
    return outliers# Driver code
sample_outliers = detect_outliers_iqr(dataset.iloc[:,[5]].values)
sample_outliers2 = detect_outliers_iqr(dataset['Precipitacion'])

print("Outliers from IQR method: ", sample_outliers)
print("Outliers from IQR method: ", sample_outliers2)
