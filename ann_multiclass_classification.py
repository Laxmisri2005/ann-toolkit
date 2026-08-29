
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
os.makedirs(
    "results",
    exist_ok=True
)


print("\nLoading CIFAR-100 dataset...")

(X_train, y_train), (X_test, y_test) = cifar100.load_data(
    label_mode="fine"
)

print("\nDataset Loaded Successfully!")


print("\n==============================")
print("DATASET INFORMATION")
print("==============================")

print(
    "Training Images:",
    X_train.shape
)

print(
    "Testing Images:",
    X_test.shape
)

print(
    "Training Labels:",
    y_train.shape
)

print(
    "Testing Labels:",
    y_test.shape
)


y_train = y_train.flatten()
y_test = y_test.flatten()

num_classes = len(
    np.unique(y_train)
)

print(
    "Number of Classes:",
    num_classes
)


class_names = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm"
]


plt.figure(
    figsize=(10, 6)
)

for i in range(10):

    plt.subplot(
        2,
        5,
        i + 1
    )

    plt.imshow(
        X_train[i]
    )

    plt.title(
        class_names[y_train[i]]
    )

    plt.axis(
        "off"
    )

plt.tight_layout()

plt.savefig(
    "results/cifar100_sample_images.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


X_train = X_train.astype(
    "float32"
) / 255.0

X_test = X_test.astype(
    "float32"
) / 255.0


X_train = X_train.reshape(
    X_train.shape[0],
    -1
)

X_test = X_test.reshape(
    X_test.shape[0],
    -1
)


print(
    "\nTraining Shape:",
    X_train.shape
)

print(
    "Testing Shape:",
    X_test.shape
)


model = Sequential()

model.add(
    Input(
        shape=(
            X_train.shape[1],
        )
    )
)

model.add(
    Dense(
        256,
        activation="relu"
    )
)

model.add(
    Dense(
        128,
        activation="relu"
    )
)

model.add(
    Dense(
        64,
        activation="relu"
    )
)

model.add(
    Dense(
        num_classes,
        activation="softmax"
    )
)


model.summary()


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print(
    "\nModel compiled successfully!"
)


history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "ANN Multiclass Classification Accuracy - CIFAR-100"
)

plt.legend()

plt.grid()

plt.savefig(
    "results/multiclass_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "ANN Multiclass Classification Loss - CIFAR-100"
)

plt.legend()

plt.grid()

plt.savefig(
    "results/multiclass_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "\nGenerating predictions..."
)

y_probability = model.predict(
    X_test
)

y_pred = np.argmax(
    y_probability,
    axis=1
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    "\n=========================================="
)

print(
    "ANN MULTICLASS CLASSIFICATION RESULTS"
)

print(
    "DATASET: CIFAR-100"
)

print(
    "=========================================="
)

print(
    "\nAccuracy:",
    accuracy
)

print(
    "\nAccuracy Percentage:",
    accuracy * 100,
    "%"
)


print(
    "\n=========================================="
)

print(
    "CLASSIFICATION REPORT"
)

print(
    "=========================================="
)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names
    )
)


cm = confusion_matrix(
    y_test,
    y_pred
)


print(
    "\n=========================================="
)

print(
    "CONFUSION MATRIX"
)

print(
    "=========================================="
)

print(
    cm
)


plt.figure(
    figsize=(16, 14)
)

plt.imshow(
    cm
)

plt.title(
    "ANN Multiclass Classification Confusion Matrix - CIFAR-100"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.xticks(
    range(num_classes),
    class_names,
    rotation=90,
    fontsize=6
)

plt.yticks(
    range(num_classes),
    class_names,
    fontsize=6
)

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "results/multiclass_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


model.save(
    "results/cifar100_ann_model.keras"
)


print(
    "\nModel saved successfully!"
)

print(
    "\n=========================================="
)

print(
    "PROGRAM COMPLETED SUCCESSFULLY"
)

print(
    "=========================================="
)

print(
    "\nResults saved inside the 'results' folder."
)

print(
    "1. cifar100_sample_images.png"
)

print(
    "2. multiclass_accuracy.png"
)

print(
    "3. multiclass_loss.png"
)

print(
    "4. multiclass_confusion_matrix.png"
)

print(
    "5. cifar100_ann_model.keras"
)

