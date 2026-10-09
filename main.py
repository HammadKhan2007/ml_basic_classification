import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("Loading Iris dataset...")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_names'] = df['target'].map({i: name for i, name in enumerate(iris.target_names)})

    print("Basic Data Exploration:")
    print(df.head())

    # Create visualization
    print("Generating pairplot visualization...")
    sns.set_theme(style="ticks")
    plot = sns.pairplot(df, hue='target_names')
    os.makedirs('outputs', exist_ok=True)
    plot.figure.savefig('outputs/pairplot.png')
    print("Saved visualization to outputs/pairplot.png")

    # Train model
    X = df.drop(['target', 'target_names'], axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

if __name__ == "__main__":
    main()
