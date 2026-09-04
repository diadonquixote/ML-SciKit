import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Dataset
data = {
    "Order": [1, 2, 3, 4, 5, 6, 7, 8],
    "Age": [21, 19, 17, 16, 14, 13, 10, 9],
}
df = pd.DataFrame(data)

x = df[["Order"]]
y = df["Age"]

# 2. Model Training
X_Train, X_Test, y_Train, y_Test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_Train, y_Train)

# 3. Create and Display Plot
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='Actual Siblings', s=60)

# Generate trendline from sibling 1 to 9
x_line = pd.DataFrame({"Order": np.linspace(1, 9, 100)})
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color='red', linestyle='--', label='Regression Line')

# Predict and highlight 9th sibling
pred_9th = model.predict(pd.DataFrame({"Order": [9]}))[0]
plt.scatter([9], [pred_9th], color='green', marker='*', s=180, label=f'9th Sibling ({pred_9th:.2f} yrs)')

plt.title('Sibling Order vs. Age', fontsize=14)
plt.xlabel('Sibling Order (x)', fontsize=12)
plt.ylabel('Age (y)', fontsize=12)
plt.xticks(range(1, 10))
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()

# 4. Interactive Terminal Prompt
print("\n Age Predictor")
print(" Type 'exit' to end the program. \n")

while True:
    user_input = input("Enter the order number: ")
    if user_input.lower() in ["exit"]:
        print("Exiting Program. Enjoy your day!")
        break

    try:
        order_number = float(user_input)
        input_data = pd.DataFrame({"Order": [order_number]})
        
        predicted_age = model.predict(input_data)[0]
        print(f"-> Predicted Age: {predicted_age:.2f}\n")
    except Exception as e:
        print(f"Error making prediction: {e}\n")