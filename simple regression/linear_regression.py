import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Prepare data
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Test_Score': [52, 55, 61, 64, 70, 74, 82, 85, 91, 96]
}
df = pd.DataFrame(data)

X = df[['Hours_Studied']]
y = df['Test_Score']

# 2. Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 3. Interactive Terminal Loop
print("\n=== Model Ready ===")
print("Type 'exit' or 'quit' to stop the program.\n")

while True:
    user_input = input("Enter hours studied: ")
    
    # Check if user wants to exit
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting program. Happy studying!")
        break
    
    try:
        # Convert input to float
        hours = float(user_input)
        
        # Scikit-learn expects 2D data: DataFrame or 2D NumPy array
        input_data = pd.DataFrame({'Hours_Studied': [hours]})
        predicted_score = model.predict(input_data)[0]
        
        print(f"-> Predicted Test Score: {predicted_score:.2f}\n")
    except ValueError:
        print("Please enter a valid number (e.g., 5 or 7.5).\n")