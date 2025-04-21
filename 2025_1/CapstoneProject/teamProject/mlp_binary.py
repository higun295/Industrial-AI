import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# 🔧 JSON 데이터 경로
json_dir = r"D:\Data\FinalData_Without_Confidence"

X_all = []
y_all = []

# ✅ JSON 파일 순회하며 데이터 수집
for file in os.listdir(json_dir):
    if file.endswith(".json"):
        with open(os.path.join(json_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                keypoints = item.get("keypoints", [])
                label = item.get("label", None)
                if not keypoints or label is None:
                    continue
                # 🔁 라벨 매핑
                if label in [1, 2, 3, 4, 5, 11]:
                    mapped_label = "Fall"
                elif label in [6, 7, 8, 9, 10, 20]:
                    mapped_label = "Not-Fall"
                else:
                    continue
                X_all.append(keypoints)
                y_all.append(mapped_label)

print("🔍 라벨 분포:", pd.Series(y_all).value_counts().sort_index())
print(len(y_all))

# 📦 배열로 변환
X = np.array(X_all)
y = np.array(y_all)

# 📚 데이터 분할 (학습:테스트 = 7:3)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)

# 🧠 MLP 모델 정의
clf = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='sgd',
    alpha=1e-8,
    batch_size=min(200, len(X_train)),
    learning_rate_init=0.001,
    max_iter=10,
    shuffle=True,
    tol=0.0001,
    momentum=0.9,
    n_iter_no_change=10,
    verbose=True
)

# 🚀 모델 학습
clf.fit(X_train, y_train)

# 🔍 예측
y_pred = clf.predict(X_test)

# 📊 confusion matrix 및 지표 계산
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# 🎨 Confusion matrix 시각화
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Fall", "Fall"],
            yticklabels=["Non-Fall", "Fall"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) != 0 else 0
recall = tp / (tp + fn) if (tp + fn) != 0 else 0
specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

# ✅ 결과 출력
print("\n🧪 [MLP Classification 수식 기반 평가 결과]")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"F1 Score     : {f1:.4f}")