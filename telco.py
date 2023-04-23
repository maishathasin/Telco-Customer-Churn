import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import make_column_transformer
from sklearn.metrics import (RocCurveDisplay, accuracy_score, auc,
                             classification_report, f1_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class Dataset:
    """
    Creates a unified way to pre-process, split, and evaluate models on the IBM
    Telco dataset.
    """
    def __init__(self, excel="Telco_customer_churn.xlsx", onehot=False,
                 scale=False, smote=False):
        # Load data
        df = pd.read_excel(excel)
        self.scaler = MinMaxScaler()

        # Remove unrelated columns
        df.drop(columns=[
            "CustomerID", "Count", "Country", "State", "City", "Zip Code",
            "Lat Long", "Latitude", "Longitude", "Churn Label", "Churn Score",
            "CLTV", "Churn Reason"
        ], inplace=True)

        # Remove rows with empty values
        df = df.loc[df["Total Charges"].str.strip() != ""]

        # Numeric columns
        num_cols = {
            "Tenure Months": int,
            "Monthly Charges": float,
            "Total Charges": float
        }
        df = df.astype(num_cols)

        # Isolate Features and Response
        X = df.drop(columns=["Churn Value"])
        Y = df["Churn Value"]

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, Y, test_size=0.1, random_state=441
        )

        # SMOTE (Synthetic Minority Over-sampling TEchnique)
        if smote:
            cat_mask = [not col in num_cols for col in X.columns]
            pipeline = make_pipeline(
                SMOTENC(cat_mask, sampling_strategy=0.5, random_state=441),
                RandomUnderSampler(sampling_strategy=0.5, random_state=441)
            )
            self.X_train, self.y_train = pipeline.fit_resample(
                self.X_train, self.y_train
            )

        # One-hot encoding
        if onehot:
            self.X_train = pd.get_dummies(self.X_train, drop_first=True)
            self.X_test = pd.get_dummies(self.X_test, drop_first=True)
            # cat_cols = [col for col in X.columns if col not in num_cols]
            # transformer = make_column_transformer(
            #     (OneHotEncoder(drop='if_binary'), cat_cols),
            #     remainder='passthrough'
            # )
            # self.X_train = pd.DataFrame(
            #     transformer.fit_transform(self.X_train),
            #     columns=transformer.get_feature_names_out()
            # )
            # self.X_test = pd.DataFrame(
            #     transformer.fit_transform(self.X_test),
            #     columns=transformer.get_feature_names_out()
            # )

        # Scale numeric columns to [0, 1]
        if scale:
            num_col_names = list(num_cols.keys())
            self.X_train[num_col_names] = self.scaler.fit_transform(
                self.X_train[num_col_names]
            )
            self.X_test[num_col_names] = self.scaler.fit_transform(
                self.X_test[num_col_names]
            )
        

    def get_training_set(self):
        return self.X_train, self.y_train

    def get_testing_set(self):
        return self.X_test

    def accuracy(self, y_pred, train=False):
        if train:
            return accuracy_score(self.y_train, y_pred)
        else:
            return accuracy_score(self.y_test, y_pred)

    def f1(self, y_pred, train=False):
        if train:
            return f1_score(self.y_train, y_pred)
        else:
            return f1_score(self.y_test, y_pred)
    
    def acc_f1(self, y_pred, train=False):
        y_pred = [0 if y < 0.5 else 1 for y in y_pred]
        return (
            round(self.accuracy(y_pred, train=train), 4),
            round(self.f1(y_pred, train=train), 4)
        )

    def report(self, y_pred):
        print(classification_report(self.y_test, y_pred))

    
    def roc_curve(self, csv_dict):
        """
        csv_dict: keys are filepaths and values are display names
        """
        for filename, display_name in csv_dict.items():
            df = pd.read_csv(filename)
            y_pred = df["y_pred"].tolist()
            fpr, tpr, _ = roc_curve(self.y_test, y_pred)
            roc_auc = round(auc(fpr, tpr), 3)
            plt.plot(fpr, tpr, label=f"{display_name}, auc="+str(roc_auc))

        plt.legend(loc=0)
        plt.show()

    def save_predictions(self, filename, y_pred):
        pd.DataFrame(y_pred, columns=["y_pred"]) \
            .to_csv(f"{filename}.csv", index=False)


if __name__ == "__main__":
    ds = Dataset()

    ds.roc_curve({
        "dtree.csv": "Decision Tree",
        "dtree+sm.csv": "Decision Tree with SMOTE",
        "rf.csv": "Random Forest",
        "rf+sm.csv": "Random Forest with SMOTE",
        "slp.csv": "Single-Layer Perceptron",
        "mlp.csv": "Multi-Layer Perceptron"
    })