import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_excel('HousePricePrediction.xlsx')

print(dataset.head(5))

dataset.shape