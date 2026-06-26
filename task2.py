import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Create Dataset
data = {
    "Hours": [1,2,3,4,5,6,7,8,9,10],
    "Score": [15,25,40,50,60,65,75,82,88,95]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['Hours']]
y = df['Score']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Output
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
