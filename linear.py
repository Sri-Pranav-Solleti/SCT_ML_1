import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("Hours_Scores.csv")  
print(data.head())

x = data[['Hours']]
y = data['Scores']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
print(f"Regression Equation: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

plt.scatter(x, y, color='blue', label='Actual Data')

x_range = pd.DataFrame(np.linspace(x['Hours'].min(), x['Hours'].max(), 100), columns=['Hours'])

plt.plot(x_range, model.predict(x_range), color='red', label='Regression Line')

plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Hours vs Scores')
plt.legend()
plt.grid(True)
plt.show()


