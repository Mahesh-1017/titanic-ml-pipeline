import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataValidationError(Exception):
    pass


def load_data():
    data = sns.load_dataset("titanic")
    return data


def validate_data(data):
    required_columns = [
        "survived",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked",
        "class"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {missing_columns}"
        )

    print("Dataset validation successful.")
    return True


def clean_text(data):
    data = data.copy()

    text_columns = ["sex", "embarked", "class"]

    for column in text_columns:
        data[column] = (
            data[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    print("Text columns cleaned.")
    return data



def drop_useless(data):
    data = data.copy()

    columns_to_drop = [
        "alive",      
        "deck",       
        "adult_male", 
        "who",         
        "embark_town", 
        "pclass"      
    ]

    data = data.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    print("Unnecessary columns removed.")
    return data


def handle_missing(data):
    data = data.copy()

    numeric_columns = data.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        if column != "survived" and data[column].isnull().any():
            data[column] = data[column].fillna(
                data[column].median()
            )

    categorical_columns = data.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in categorical_columns:
        if data[column].isnull().any():
            mode_values = data[column].mode()

            if not mode_values.empty:
                data[column] = data[column].fillna(
                    mode_values.iloc[0]
                )

    print("Missing values handled.")
    return data


def encode_categoricals(data):
    data = data.copy()

 
    data["sex"] = data["sex"].map({
        "male": 0,
        "female": 1
    })

    data["class"] = data["class"].map({
        "first": 1,
        "second": 2,
        "third": 3
    })

    
    data = pd.get_dummies(
        data,
        columns=["embarked"],
        prefix="embarked",
        dtype=int
    )

    print("Categorical columns encoded.")
    return data


def split_data(data):
    X = data.drop(columns=["survived"])
    y = data["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Dataset split completed.")
    return X_train, X_test, y_train, y_test



def scale_features(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()

    scale_columns = [
        "age",
        "sibsp",
        "parch",
        "fare"
    ]

   
    X_train[scale_columns] = scaler.fit_transform(
        X_train[scale_columns]
    )

    X_test[scale_columns] = scaler.transform(
        X_test[scale_columns]
    )

    print("Numerical features scaled.")
    return X_train, X_test


def main():
    try:
        data = load_data()
        print("Original dataset shape:", data.shape)

      
        validate_data(data)
        data = clean_text(data)
        data = drop_useless(data)
        data = handle_missing(data)
        data = encode_categoricals(data)

        X_train, X_test, y_train, y_test = split_data(data)

        X_train, X_test = scale_features(
            X_train,
            X_test
        )

        print("\n--- Pipeline Completed Successfully ---")
        print("Training features:", X_train.shape)
        print("Testing features:", X_test.shape)
        print("Training target:", y_train.shape)
        print("Testing target:", y_test.shape)

        print("\nFirst five rows of training features:")
        print(X_train.head())

        print("\nMissing values in training features:")
        print(X_train.isnull().sum())

        return X_train, X_test, y_train, y_test

    except DataValidationError as error:
        print("Data validation error:", error)

    except Exception as error:
        print("Pipeline failed:", error)



if __name__ == "__main__":
    result = main()
