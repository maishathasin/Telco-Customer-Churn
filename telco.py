from collections import namedtuple

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd


class Dataset:
    def __init__(self, excel="Telco_customer_churn.xlsx"):
        # Load data
        df = pd.read_excel(excel)

        # Remove unrelated columns
        df.drop(columns=[
            "CustomerID", "Count", "Country", "State", "City", "Zip Code",
            "Lat Long", "Latitude", "Longitude", "Churn Label", "Churn Score",
            "CLTV", "Churn Reason"
        ], inplace=True)

        # Remove rows with empty values
        df = df.loc[df["Total Charges"].str.strip() != ""]
        df['Total Charges'] = df['Total Charges'].astype('float')

        # Isolate Features and Response
        X = df.drop(columns=["Churn Value"])
        Y = df["Churn Value"]
        onehot_X = pd.get_dummies(X)

        # Remove evaluation set
        SplitData = namedtuple("SplitData", [
            "X", "X_eval", "ohX", "ohX_eval", "y", "y_eval"
        ])
        self._data = SplitData(*train_test_split(
            X, onehot_X, Y, test_size=0.1, random_state=441
        ))

    def get_training_set(self, onehot=False):
        if onehot:
            return self._data.ohX, self._data.y
        else:
            return self._data.X, self._data.y

    def get_testing_set(self, onehot=False):
        if onehot:
            return self._data.ohX_eval
        else:
            return self._data.X_eval

    def accuracy(self, y_pred):
        return accuracy_score(self._data.y_eval, y_pred)
