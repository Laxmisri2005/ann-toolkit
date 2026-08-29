import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


os.makedirs("results", exist_ok=True)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


df = pd.read_csv("Sales - Marketing customer dataset.csv")

section("DATASET OVERVIEW")

print(df.head())

print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

print(f"Missing values: {df.isnull().sum().sum()} total")



section("TARGET DISTRIBUTION (churn)")

counts = df["churn"].value_counts()
total = len(df)
for label, name in [(0, "No Churn"), (1, "Churn")]:
    n = counts.get(label, 0)
    print(f"  {name:<10} {n:>6}  ({n/total*100:5.1f}%)")




df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"], errors="coerce")

reference_date = df["last_purchase_date"].max()

df["tenure_days"] = (reference_date - df["signup_date"]).dt.days
df["days_since_last_purchase"] = (reference_date - df["last_purchase_date"]).dt.days


drop_columns = [
    "customer_id",
    "signup_date",
    "last_purchase_date",
    "coupon_code",
    "churn"
]

X = df.drop(
    columns=drop_columns,
    errors="ignore"
)

y = df["churn"]


categorical_columns = X.select_dtypes(
    include=["object"]
).columns

numeric_columns = X.select_dtypes(
    include=["number"]
).columns


# Impute missing values: median for numeric columns, "Unknown" for categorical.
for col in numeric_columns:
    X[col] = X[col].fillna(X[col].median())

for col in categorical_columns:
    X[col] = X[col].fillna("Unknown")


X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

X = X.astype(float)


X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


section("FEATURE PREPARATION")

print(f"Processed feature matrix: {X.shape[0]} rows x {X.shape[1]} columns")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print(f"Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")

print(f"Testing:  {X_test.shape[0]} samples, {X_test.shape[1]} features")


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# The target is imbalanced (~15% churn), so weight classes to avoid the
# model just predicting "no churn" for everyone.
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(enumerate(class_weights))

print("\nClass weights:", class_weight_dict)


model = Sequential()

model.add(Input(shape=(X_train.shape[1],)))


model.add(
    Dense(
        64,
        activation="relu"
    )
)


model.add(
    Dense(
        32,
        activation="relu"
    )
)


model.add(
    Dense(
        16,
        activation="relu"
    )
)


model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)


model.summary()


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=128,
    validation_split=0.2,
    class_weight=class_weight_dict,
    verbose=1
)


plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "ANN Binary Classification Accuracy (Customer Churn)"
)

plt.legend()

plt.grid()

plt.savefig(
    "results/binary_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)


plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "ANN Binary Classification Loss (Customer Churn)"
)

plt.legend()

plt.grid()

plt.savefig(
    "results/binary_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)


y_probability = model.predict(
    X_test
)


y_pred = (
    y_probability >= 0.5
).astype(int).flatten()


accuracy = accuracy_score(
    y_test,
    y_pred
)


section("ANN BINARY CLASSIFICATION RESULTS (Customer Churn)")

print(f"Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)")

print("\nClassification Report:")


print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Churn",
            "Churn"
        ]
    )
)


cm = confusion_matrix(
    y_test,
    y_pred
)


print("\nConfusion Matrix:")
print(f"{'':>12}{'Pred: No':>10}{'Pred: Churn':>13}")
print(f"{'Actual: No':>12}{cm[0][0]:>10}{cm[0][1]:>13}")
print(f"{'Actual: Churn':>12}{cm[1][0]:>10}{cm[1][1]:>13}")


plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title(
    "ANN Binary Classification Confusion Matrix (Customer Churn)"
)


plt.xticks(
    [0, 1],
    ["No Churn", "Churn"]
)


plt.yticks(
    [0, 1],
    ["No Churn", "Churn"]
)


for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white")


plt.colorbar()


plt.savefig(
    "results/binary_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nSaved plots to results/binary_accuracy.png, results/binary_loss.png, results/binary_confusion_matrix.png")
print("Close the plot windows to end the script.")

plt.show()
