from sklearn.metrics import recall_score, precision_score, f1_score


def display_results(y_true, y_pred, title):
    # Hàm hỗ trợ in kết quả
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"--- {title} ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")
