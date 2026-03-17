import pandas as pd
from sklearn.linear_model import LinearRegression

from cleaner import HouseCleaner


def main() -> None:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    X_train = train.drop(columns=["SalePrice"])
    y_train = train["SalePrice"]

    cleaner = HouseCleaner()
    X_train_clean = cleaner.fit_transform(X_train)
    X_test_clean = cleaner.transform(test)

    # Keep all test rows and align feature columns safely by name.
    feature_names = [f"f{i}" for i in range(X_train_clean.shape[1])]
    X_train_df = pd.DataFrame(X_train_clean, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_clean, columns=feature_names, index=test.index)

    model = LinearRegression()
    model.fit(X_train_df, y_train)
    preds = model.predict(X_test_df)

    submission = pd.DataFrame({"Id": test["Id"], "SalePrice": preds})

    expected_rows = len(test)
    if len(submission) != expected_rows:
        raise ValueError(
            f"Submission row mismatch: got {len(submission)} rows, expected {expected_rows}."
        )

    submission.to_csv("submission_lr.csv", index=False)
    print(f"Saved submission_lr.csv with {len(submission)} rows.")


if __name__ == "__main__":
    main()
