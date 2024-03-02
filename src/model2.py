from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

iris = load_iris()
X, y = iris.data, iris.target

# data prep
X_train, X_test, y_train, y_test = train_test_split(X, y)

# training
clr = RandomForestClassifier()
clr.fit(X_train, y_train)

print(clr)

# export to as pickl file
joblib.dump(clr, "../output/model2.pkl", compress=9)