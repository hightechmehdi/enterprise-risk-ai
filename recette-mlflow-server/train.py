import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os


# Load Iris dataset
iris = load_iris()

# Split dataset into X features and Target variable
X = pd.DataFrame(data = iris["data"], columns= iris["feature_names"])
y = pd.Series(data = iris["target"], name="target")

# Split our training set and our test set
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Visualize dataset
X_train.head()


# Set your variables for your environment
EXPERIMENT_NAME=os.environ["EXPERIMENT_NAME"]
MODEL_NAME = "recette-mlflow-server"

# Set tracking URI to your Hugging Face application
mlflow.set_tracking_uri(os.environ["APP_URI"])

# Set experiment's info
mlflow.set_experiment(EXPERIMENT_NAME)

# Get our experiment info
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

# Call mlflow autolog
mlflow.sklearn.autolog(log_models=False)

with mlflow.start_run(experiment_id = experiment.experiment_id):
    # Specified Parameters
    c = 0.5

    # Instanciate and fit the model
    lr = LogisticRegression(C=c)
    lr.fit(X_train.values, y_train.values)

    # Store metrics
    predicted_qualities = lr.predict(X_test.values)
    accuracy = lr.score(X_test.values, y_test.values)

    mlflow.sklearn.log_model(
        lr,
        name="model",
        signature=infer_signature(X_train, predicted_qualities),
        input_example=X_train.head(2),
        registered_model_name=MODEL_NAME,
        serialization_format="cloudpickle"
    )

    # Print results
    print("LogisticRegression model")
    print("Accuracy: {}".format(accuracy))

client = mlflow.MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
last_version = max(int(v.version) for v in versions)
client.set_registered_model_alias(MODEL_NAME, "recette-champion", last_version)
print(f"Modèle '{MODEL_NAME}' version {last_version} enregistré avec l'alias @recette-champion")
