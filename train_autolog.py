import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

iris = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')

X = iris.iloc[:,0:-1]
y = iris.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the Random Forest model
max_depth = 1
n_estimators = 100

# apply mlflow
mlflow.autolog()

mlflow.set_experiment('iris-rf')

with mlflow.start_run():

    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)


    # Create a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    # Save the plot as an artifact
    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact(__file__)

    mlflow.set_tag('author','suryansh')
    mlflow.set_tag('model','random forest')

    print('accuracy', accuracy)