import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


os.makedirs("results", exist_ok=True)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


df = pd.read_csv("robot_inverse_kinematics_dataset.csv")

section("DATASET OVERVIEW")

print(df.head())

print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

print(f"Missing values: {df.isnull().sum().sum()} total")


# This is a robot INVERSE KINEMATICS dataset: q1..q6 are joint angles and
# x, y, z is the end-effector position produced by those angles (forward
# kinematics). The inverse-kinematics problem is the reverse mapping: given
# a target end-effector position (x, y, z), predict the joint angles
# (q1..q6) that reach it. There is no "price" column in this dataset, so
# the model below predicts all 6 joint angles from the 3 position values.

feature_columns = ["x", "y", "z"]
target_columns = ["q1", "q2", "q3", "q4", "q5", "q6"]

X = df[feature_columns]

y = df[target_columns]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


section("TRAIN / TEST SPLIT")

print(f"Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")

print(f"Testing:  {X_test.shape[0]} samples, {X_test.shape[1]} features")


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Targets (joint angles) also benefit from scaling since they have
# different ranges; we invert this scaling before computing metrics.
y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(y_train)

y_test_scaled = y_scaler.transform(y_test)


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
        len(target_columns),
        activation="linear"
    )
)


model.summary()


model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


history = model.fit(
    X_train,
    y_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


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

plt.ylabel("MSE (scaled targets)")

plt.title(
    "ANN Training and Validation Loss"
)

plt.legend()

plt.grid()

plt.savefig(
    "results/regression_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)


y_pred_scaled = model.predict(
    X_test
)

# Undo target scaling to get predictions back in radians (original units).
y_pred = y_scaler.inverse_transform(y_pred_scaled)

y_test_arr = y_test.to_numpy()


mae = mean_absolute_error(
    y_test_arr,
    y_pred
)


mse = mean_squared_error(
    y_test_arr,
    y_pred
)


rmse = np.sqrt(mse)


r2 = r2_score(
    y_test_arr,
    y_pred
)


section("ANN REGRESSION RESULTS (Inverse Kinematics)")

print(f"{'MAE':6}: {mae:.4f}")
print(f"{'MSE':6}: {mse:.4f}")
print(f"{'RMSE':6}: {rmse:.4f}")
print(f"{'R2':6}: {r2:.4f}")
print("  (overall metrics, averaged across all 6 joints)")

print("\nPer-joint R2 score:")
print(f"  {'Joint':<6}{'R2':>8}")
print(f"  {'-'*14}")
for i, joint in enumerate(target_columns):
    joint_r2 = r2_score(y_test_arr[:, i], y_pred[:, i])
    print(f"  {joint:<6}{joint_r2:>8.4f}")


fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for i, joint in enumerate(target_columns):

    ax = axes[i // 3, i % 3]

    ax.scatter(
        y_test_arr[:, i],
        y_pred[:, i],
        alpha=0.4,
        s=10
    )

    lims = [
        min(y_test_arr[:, i].min(), y_pred[:, i].min()),
        max(y_test_arr[:, i].max(), y_pred[:, i].max())
    ]

    ax.plot(lims, lims, "r--", linewidth=1)

    ax.set_xlabel(f"Actual {joint}")

    ax.set_ylabel(f"Predicted {joint}")

    ax.set_title(joint)

    ax.grid()

fig.suptitle("Actual vs Predicted Joint Angles (Inverse Kinematics)")

fig.tight_layout()

fig.savefig(
    "results/regression_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nSaved plots to results/regression_loss.png and results/regression_actual_vs_predicted.png")
print("Close the plot windows to end the script.")

plt.show()
