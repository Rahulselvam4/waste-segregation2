# ─────────────────────────────────────────────────────────────────
# model_builder.py
# Rebuilds the exact architecture used in training.
# Avoids ALL Keras version mismatch errors when loading weights.
# ─────────────────────────────────────────────────────────────────

import os
import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D,
    BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam

# ── Load class names from json (single source of truth) ──────────
_json_path = os.path.join(os.path.dirname(__file__), "class_names.json")

if os.path.exists(_json_path):
    with open(_json_path, 'r') as f:
        CLASS_NAMES = json.load(f)
else:
    # Fallback if json not present
    CLASS_NAMES = [
        'cardboard', 'glass', 'metal', 'paper', 'plastic',
        'trash', 'biological', 'battery', 'shoes', 'clothes'
    ]

NUM_CLASSES  = len(CLASS_NAMES)
IMG_SIZE     = (130, 130)
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__),
                             "waste_weights_FINAL.weights.h5")

_model = None  # cached model instance


def build_architecture():
    """
    Rebuilds the EXACT same architecture used in Part 2 training.
    Must match cell 4 of the training notebook exactly.

    NOTE: Dense sizes (512 → 256) were confirmed from the saved weights file.
    The error "value.shape=(1280, 512)" tells us the first Dense must be 512,
    and the second Dense must be 256 to match what was actually trained.
    """
    base_model = MobileNetV2(
        input_shape=(130, 130, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True

    inputs = Input(shape=(130, 130, 3))
    x      = base_model(inputs, training=False)
    x      = GlobalAveragePooling2D()(x)
    x      = BatchNormalization()(x)
    x      = Dense(512, activation='relu')(x)   # was 256 — fixed to match weights
    x      = Dropout(0.4)(x)
    x      = Dense(256, activation='relu')(x)   # was 128 — fixed to match weights
    x      = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_model_safe():
    """
    Loads model weights into rebuilt architecture.
    Cached after first load — no repeated disk reads.
    """
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"\n\nModel weights not found at:\n{WEIGHTS_PATH}\n\n"
            "Please place waste_weights_FINAL.weights.h5 "
            "in the project root folder.\n"
        )

    print("Loading model weights...")
    model = build_architecture()
    model.load_weights(WEIGHTS_PATH)
    _model = model
    print(f"Model loaded! Classes: {CLASS_NAMES}")
    return _model