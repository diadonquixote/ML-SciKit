import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Create the dataset directly inside Python
data = {
    'Hours_Studied': [2.5, 5.0, 1.5, 8.0, 4.0, 6.5, 3.0, 9.0, 5.5, 7.0],
    'Attendance_Rate': [60, 80, 50, 95, 70, 85, 65, 98, 75, 90],
    'Final_Score': [65, 78, 52, 92, 71, 88, 64, 96, 81, 89]
}

df = pd.DataFrame(data)

# Save it to CSV automatically so the file exists on your disk
df.to_csv('student_data.csv', index=False)

# STEP 2: Define 2 features for X, and 1 target for y
X = df[['Hours_Studied', 'Attendance_Rate']]
y = df['Final_Score']

# STEP 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Inspect the learned parameters
print("--- Model Learned Parameters ---")
print("Slopes (m1 for Hours, m2 for Attendance):", model.coef_)
print("Intercept (b):", model.intercept_)

print(type(model).__module__)

# STEP 5: Make a prediction
sample_student = pd.DataFrame({'Hours_Studied': [6.0], 'Attendance_Rate': [82]})
predicted_score = model.predict(sample_student)[0]
print(f"\nPredicted Final Score: {predicted_score:.2f}")