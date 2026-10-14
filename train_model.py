import json
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

def main():
    if not DATA_DIR.exists():
        raise SystemExit("Dataset folder is missing. Create dataset/Player_Name/ and add images.")
    class_folders = [p for p in DATA_DIR.iterdir() if p.is_dir() and any(
        f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'} for f in p.iterdir() if f.is_file())]
    if len(class_folders) < 2:
        raise SystemExit("Add images for at least TWO players, each in its own folder inside dataset/.")
    for folder in class_folders:
        count = sum(1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
        print(f"Found {count:>3} images: {folder.name}")
        if count < 5:
            print(f"WARNING: {folder.name} has fewer than 5 images. Use 20–50+ varied photos per player if possible.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int")
    class_names = train_ds.class_names
    print("Classes:", class_names)
    (MODEL_DIR / "labels.json").write_text(
        json.dumps({str(i): name for i, name in enumerate(class_names)}, indent=2),
        encoding="utf-8")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
                            num_parallel_calls=autotune).prefetch(autotune)
    val_ds = val_ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
                        num_parallel_calls=autotune).prefetch(autotune)

    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.06),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ], name="augmentation")

    base = MobileNetV2(input_shape=(224,224,3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = layers.Input(shape=(224,224,3))
    x = augmentation(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_DIR / "best_model.keras"),
                                           monitor="val_accuracy", save_best_only=True),
    ]
    print("\nTraining the classification head...")
    model.fit(train_ds, validation_data=val_ds, epochs=12, callbacks=callbacks)

    # Save best checkpoint if available; otherwise save current model.
    best_path = MODEL_DIR / "best_model.keras"
    if best_path.exists():
        model = tf.keras.models.load_model(best_path)
    model.save(MODEL_DIR / "cricket_face_model.keras")
    print(f"\nDone! Model saved to: {MODEL_DIR / 'cricket_face_model.keras'}")
    print("Now run: streamlit run app.py")

if __name__ == "__main__":
    main()
