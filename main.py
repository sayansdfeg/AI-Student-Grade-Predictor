import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1], [2], [3], [4], [5], [6], [7]])
y = np.array([52, 58, 65, 71, 78, 85, 92])
model = LinearRegression()
model.fit(X, y)
hours_to_predict = 8.5
predicted_grade = model.predict([[hours_to_predict]])

print(f"--- AI Grade Predictor ---")
print(f"Если учиться {hours_to_predict} часов, ИИ предсказывает оценку: {predicted_grade[0]:.2f} баллов.")
print(f"Формула модели: Оценка = {model.coef_[0]:.2f} * Часы + {model.intercept_:.2f}")
