import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("House Price dataset.csv")

X = data[['Bedrooms', 'Bathrooms', 'Size(sqft)']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))

sample_house = [[3, 2, 1500]]
predicted_price = model.predict(sample_house)

print("Predicted House Price:", predicted_price[0])
