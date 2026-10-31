import streamlit as st
import zipfile
import tempfile
import matplotlib.pyplot as plt
import numpy as np

from cnn_model import build_cnn
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, load_img, img_to_array
)
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="CNN Image Classifier",
    layout="wide"
)

st.title("CNN Image Classifier")
st.write("Train a CNN model and predict image classes.")

# CIFAR-10 class names
CIFAR_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# -----------------------------
# Session State
# -----------------------------
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.img_size = 32
    st.session_state.class_names = []
    st.session_state.trained_dataset = None

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Choose Section")

section = st.sidebar.radio(
    "Go to:",
    ["Train Model", "Predict"]
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
if section == "Train Model":

    dataset_choice = st.sidebar.radio(
        "Select Dataset:",
        ["CIFAR-10", "Cats vs Dogs (ZIP Upload)"]
    )

    batch_size = st.sidebar.selectbox(
        "Batch Size", [16, 32, 64]
    )

    epochs = st.sidebar.slider(
        "Epochs", min_value=1, max_value=20, value=5
    )

    # -----------------------------
    # CASE 1: CIFAR-10
    # -----------------------------
    if dataset_choice == "CIFAR-10":

        st.subheader("CIFAR-10 Dataset")
        st.write("This dataset contains 10 image classes.")

        st.info(
            "The CIFAR-10 dataset will download automatically "
            "the first time you use it."
        )

        if st.button("Train CNN on CIFAR-10"):

            try:
                with st.spinner("Loading dataset..."):
                    (x_train, y_train), (x_test, y_test) = (
                        cifar10.load_data()
                    )

                    x_train = x_train.astype("float32") / 255.0
                    x_test = x_test.astype("float32") / 255.0

                    y_train_cat = to_categorical(
                        y_train, num_classes=10
                    )
                    y_test_cat = to_categorical(
                        y_test, num_classes=10
                    )

                with st.spinner("Training CNN model..."):

                    model = build_cnn((32, 32, 3), 10)

                    history = model.fit(
                        x_train,
                        y_train_cat,
                        validation_data=(x_test, y_test_cat),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1
                    )

                st.session_state.model = model
                st.session_state.img_size = 32
                st.session_state.class_names = CIFAR_CLASSES
                st.session_state.trained_dataset = "CIFAR-10"

                st.success("Training completed successfully!")

                # Accuracy and Loss Graphs
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))

                ax[0].plot(
                    history.history["accuracy"],
                    label="Training Accuracy"
                )
                ax[0].plot(
                    history.history["val_accuracy"],
                    label="Validation Accuracy"
                )
                ax[0].set_title("Model Accuracy")
                ax[0].set_xlabel("Epoch")
                ax[0].legend()

                ax[1].plot(
                    history.history["loss"],
                    label="Training Loss"
                )
                ax[1].plot(
                    history.history["val_loss"],
                    label="Validation Loss"
                )
                ax[1].set_title("Model Loss")
                ax[1].set_xlabel("Epoch")
                ax[1].legend()

                st.pyplot(fig)
                plt.close(fig)

            except Exception as error:
                st.error(f"Training error: {error}")

    # -----------------------------
    # CASE 2: CATS VS DOGS
    # -----------------------------
    else:

        st.subheader("Cats vs Dogs Dataset")

        st.write(
            "Upload a ZIP file containing separate folders "
            "for each class, such as cats/ and dogs/."
        )

        img_size = st.sidebar.slider(
            "Image Size", min_value=32, max_value=256, value=64
        )

        zip_file = st.file_uploader(
            "Upload Dataset ZIP",
            type=["zip"]
        )

        if zip_file:

            if st.button("Train CNN on Uploaded Dataset"):

                try:
                    with st.spinner("Extracting dataset..."):

                        with tempfile.TemporaryDirectory() as temp_dir:

                            with zipfile.ZipFile(
                                zip_file, "r"
                            ) as zip_ref:

                                # Basic protection against unsafe paths
                                root = __import__("os").path.realpath(temp_dir)

                                for member in zip_ref.infolist():
                                    target = __import__("os").path.realpath(
                                        __import__("os").path.join(
                                            temp_dir, member.filename
                                        )
                                    )

                                    if (
                                        target != root
                                        and not target.startswith(
                                            root + __import__("os").path.sep
                                        )
                                    ):
                                        raise ValueError(
                                            "ZIP contains an unsafe file path."
                                        )

                                zip_ref.extractall(temp_dir)

                            # Find class folders, including one
                            # optional parent folder inside the ZIP
                            import os

                            dataset_dir = temp_dir

                            while True:
                                entries = [
                                    name for name in os.listdir(dataset_dir)
                                    if os.path.isdir(
                                        os.path.join(dataset_dir, name)
                                    )
                                ]

                                files = [
                                    name for name in os.listdir(dataset_dir)
                                    if os.path.isfile(
                                        os.path.join(dataset_dir, name)
                                    )
                                ]

                                if len(entries) == 1 and not files:
                                    dataset_dir = os.path.join(
                                        dataset_dir, entries[0]
                                    )
                                else:
                                    break

                            class_dirs = [
                                name for name in os.listdir(dataset_dir)
                                if os.path.isdir(
                                    os.path.join(dataset_dir, name)
                                )
                            ]

                            if len(class_dirs) < 2:
                                raise ValueError(
                                    "ZIP must contain at least two class "
                                    "folders, such as cats and dogs."
                                )

                            datagen = ImageDataGenerator(
                                rescale=1.0 / 255.0,
                                validation_split=0.2
                            )

                            train_gen = datagen.flow_from_directory(
                                dataset_dir,
                                target_size=(img_size, img_size),
                                batch_size=batch_size,
                                class_mode="categorical",
                                subset="training",
                                seed=42
                            )

                            val_gen = datagen.flow_from_directory(
                                dataset_dir,
                                target_size=(img_size, img_size),
                                batch_size=batch_size,
                                class_mode="categorical",
                                subset="validation",
                                seed=42
                            )

                            if train_gen.samples == 0 or val_gen.samples == 0:
                                raise ValueError(
                                    "Not enough images for training "
                                    "and validation."
                                )

                            with st.spinner("Training CNN model..."):

                                model = build_cnn(
                                    (img_size, img_size, 3),
                                    train_gen.num_classes
                                )

                                history = model.fit(
                                    train_gen,
                                    validation_data=val_gen,
                                    epochs=epochs,
                                    verbose=1
                                )

                            st.session_state.model = model
                            st.session_state.img_size = img_size
                            st.session_state.class_names = [
                                name for name, index in sorted(
                                    train_gen.class_indices.items(),
                                    key=lambda item: item[1]
                                )
                            ]
                            st.session_state.trained_dataset = (
                                "Cats vs Dogs"
                            )

                            st.success("Training completed successfully!")

                            fig, ax = plt.subplots(
                                1, 2, figsize=(12, 4)
                            )

                            ax[0].plot(
                                history.history["accuracy"],
                                label="Training Accuracy"
                            )
                            ax[0].plot(
                                history.history["val_accuracy"],
                                label="Validation Accuracy"
                            )
                            ax[0].set_title("Model Accuracy")
                            ax[0].legend()

                            ax[1].plot(
                                history.history["loss"],
                                label="Training Loss"
                            )
                            ax[1].plot(
                                history.history["val_loss"],
                                label="Validation Loss"
                            )
                            ax[1].set_title("Model Loss")
                            ax[1].legend()

                            st.pyplot(fig)
                            plt.close(fig)

                except Exception as error:
                    st.error(f"Dataset or training error: {error}")

# -----------------------------
# PREDICT
# -----------------------------
elif section == "Predict":

    st.subheader("Predict Image Class")

    if st.session_state.model is None:
        st.warning(
            "Please train a model first in the 'Train Model' section."
        )

    else:
        st.write(
            f"Trained dataset: "
            f"{st.session_state.trained_dataset}"
        )

        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:

            try:
                img = load_img(
                    uploaded_file,
                    target_size=(
                        st.session_state.img_size,
                        st.session_state.img_size
                    )
                )

                st.image(
                    img,
                    caption="Uploaded Image",
                    width=300
                )

                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                with st.spinner("Predicting image..."):
                    predictions = st.session_state.model.predict(
                        img_array, verbose=0
                    )

                predicted_index = int(np.argmax(predictions[0]))
                confidence = float(
                    predictions[0][predicted_index]
                ) * 100

                class_names = st.session_state.class_names

                if predicted_index < len(class_names):
                    class_name = class_names[predicted_index]

                    st.success(
                        f"Predicted Class: {class_name}"
                    )
                    st.write(
                        f"Confidence: {confidence:.2f}%"
                    )
                else:
                    st.error("Prediction class was not found.")

            except Exception as error:
                st.error(f"Prediction error: {error}")