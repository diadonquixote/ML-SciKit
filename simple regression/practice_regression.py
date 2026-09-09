import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "Temperature (°C)": [20, 22, 25, 28, 30, 32],
    "Sales ($)": [150, 180, 220, 260, 290, 320]
}
df = pd.DataFrame(data)

X = df[["Temperature (°C)"]]
y = df["Sales ($)"]

X_Train, X_Test, y_Train, y_Test    = train_test_split(X, y, test_size=0.2, random_state=42)    
model = LinearRegression()
model.fit(X_Train, y_Train)

print("\n Ice Cream Sales Predicter")
print("\n Type 'exit' to end the program. \n")

while True:
    user_input = input("Enter the temperature in Celsius: ")
    if user_input.lower() in ["exit"]:
        print("Exiting Program. Enjoy your Ice Cream!")
        break

    try: temperature = float(user_input)
    
        input_data = pd.DataFrame({"Temperature (°C)": [temperature]})
        predicted_sales = model
        .predict(input_data)[0]
        print(f"-> Predicted Ice Cream Sales: ${predicted_sales:.2f}\n")
    except ValueError:
        print("Invalid input. Please enter a valid temperature.")
    except Exception as e:
        print(f"An error occurred: {e}")