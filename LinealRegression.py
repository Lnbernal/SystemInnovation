import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = {
    "Training Hours": [5, 8, 6, 4, 9, 3, 10, 7, 6, 8, 5, 2, 11, 1, 12, 3, 9, 2, 13, 1],
    "Balanced Diet": [7, 8, 6, 5, 9, 4, 9, 7, 6, 8, 5, 3, 10, 2, 10, 4, 9, 2, 10, 1],
    "Performance":   [7.5, 8.2, 7.0, 6.0, 9.0, 5.0, 9.5, 7.8, 7.2, 8.5, 6.8, 4.0, 9.8, 3.0, 10, 5.2, 9.1, 3.5, 10, 2.5]
}

df = pd.DataFrame(data)

x = df[["Training Hours", "Balanced Diet"]]
y = df[["Performance"]]


model = LinearRegression()
model.fit(x,y)

def Rendimiento(hours, diet):
    result = model.predict([[hours, diet]])[0][0]
    return result