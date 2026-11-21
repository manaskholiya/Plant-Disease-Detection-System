import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Cell 1: Environment Setup
try:
    import seaborn as sns
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[INFO] TensorFlow/Seaborn not installed yet. Install via requirements.txt to run full training.")

# Cell 2: Dataset Path Configuration & Hyperparameters (Matches Partial Code Implementation.pdf)
dataset_path = "Dataset1"
train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "valid")
image_size = (128, 128)
batch_size = 32
seed = 42

def run_pipeline():
    # Cell 3: Exploratory Data Analysis (EDA) - Class Frequency Extraction
    classes = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    class_counts = {c: len(os.listdir(os.path.join(train_path, c))) for c in classes}
    eda_df = pd.DataFrame(list(class_counts.items()), columns=["Disease_Class", "Count"])
    print(f"Total Classes: {len(eda_df)}")
    eda_df = eda_df.sort_values(by="Count", ascending=False)
    print(f"Total Images: {eda_df['Count'].sum()}")
    print(eda_df.head(10))

    # Cell 4: Class Distribution Plotting (Saved to static/eda_distribution.png)
    plt.figure(figsize=(14, 8))
    plt.barh(eda_df["Disease_Class"][:15], eda_df["Count"][:15], color="#2e8b57")
    plt.title("Dataset Class Distribution (EDA)")
    plt.xlabel("Number of Images")
    plt.ylabel("Classes")
    plt.tight_layout()
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/eda_distribution.png")
    plt.close()

    if not TF_AVAILABLE:
        return

    # Cell 5: Dataset Loading & Categorical Label Encoding
    train_set = tf.keras.utils.image_dataset_from_directory(
        train_path,
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )
    validation_set = tf.keras.utils.image_dataset_from_directory(
        valid_path,
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )
    class_names = train_set.class_names

    # Cell 7: Data Augmentation, Rescaling & IO Prefetching
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1)
    ])
    autotune = tf.data.AUTOTUNE
    train_set = train_set.prefetch(buffer_size=autotune)
    validation_set = validation_set.prefetch(buffer_size=autotune)

    # Architecture Pipeline (Matches PHOTO-2026-10-11-21-49-53.jpg)
    # 128x128 RGB -> Conv2D+ReLU -> MaxPooling2D -> Dropout -> Flatten -> Dense -> Softmax (38 Outputs)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        data_augmentation,
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(len(class_names), activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    epochs = int(os.environ.get("TRAIN_EPOCHS", 1))
    model.fit(train_set, validation_data=validation_set, epochs=epochs)
    model.save("trained_plant_disease_model.keras")
    print("[SUCCESS] Saved serialized model to trained_plant_disease_model.keras")

if __name__ == "__main__":
    run_pipeline()
