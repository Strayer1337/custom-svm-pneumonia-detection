from sklearn.svm import SVC

from pneumonia_detection.data_loader import load_data, TRAIN_DIR, VAL_DIR, TEST_DIR
from pneumonia_detection.soft_margin_svm import SoftMarginSVM
from pneumonia_detection.evaluator import display_results


def main():
    # Load dữ liệu
    X_train, y_train = load_data(TRAIN_DIR)
    X_val, y_val = load_data(VAL_DIR)
    X_test, y_test = load_data(TEST_DIR)

    print(f"Kích thước tập Train: {X_train.shape}")
    print(f"Kích thước tập Validation: {X_val.shape}")
    print(f"Kích thước tập Test: {X_test.shape}\n")

    # --- Assignment 1: Custom Soft-Margin SVM ---
    custom_svm = SoftMarginSVM(lr=1e-5, lambda_param=0.01, n_iters=500)
    custom_svm.fit(X_train, y_train)

    y_pred_custom = custom_svm.predict(X_val)
    display_results(y_val, y_pred_custom, "Custom SVM Results")

    # --- Assignment 2: Sklearn SVM ---
    sklearn_svm = SVC(kernel='linear', C=0.01)
    sklearn_svm.fit(X_train, y_train)

    y_pred_sklearn = sklearn_svm.predict(X_val)
    display_results(y_val, y_pred_sklearn, "Sklearn SVM Results")

    # --- Đánh giá cuối trên tập Test ---
    y_pred_test = custom_svm.predict(X_test)
    display_results(y_test, y_pred_test, "Final Custom SVM - Test Set Performance")


if __name__ == "__main__":
    main()
