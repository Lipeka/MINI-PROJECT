from google.colab import files
files.upload() #upload your kaggle.json file 
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!pip install -q kaggle
!kaggle datasets download -d shubhamgoel27/dermnet
!unzip dermnet.zip -d dermnet
!pip install -q tensorflow keras tensorflow-addons tf-keras-vis kagglehub ultralytics requests
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import shutil
import glob
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from ultralytics import YOLO
import requests
from matplotlib import pyplot as plt

# Configure GPU
print("\n🔧 Checking for GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"⚠️ RuntimeError: {e}")
else:
    print("⚠️ No GPU found")

# === STEP 1: Dataset Path ===
base_path = "/content/dermnet"
print(f"\n📁 Using dataset from {base_path}")

# === STEP 2: Data Preparation ===
IMG_SIZE = 224
BATCH_SIZE = 16
initial_epochs = 15
fine_tune_epochs = 10

print("\n🔄 Preparing data generators...")
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.4,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    base_path, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical', subset='training')

val_gen = datagen.flow_from_directory(
    base_path, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')

# === STEP 3: Build and Train MobileNetV2 Classifier ===
print("\n🧠 Building MobileNetV2 model...")
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(train_gen.num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

class_weights = dict(enumerate(class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)))

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=2)
]

print("\n🚀 Starting initial training...")
model.fit(train_gen, epochs=initial_epochs, validation_data=val_gen,
          callbacks=callbacks, class_weight=class_weights)

print("\n🔧 Fine-tuning the model...")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_gen, epochs=initial_epochs + fine_tune_epochs,
          initial_epoch=initial_epochs, validation_data=val_gen,
          callbacks=callbacks, class_weight=class_weights)

model.save("skin_gradcam_safe.keras")
print("✅ MobileNetV2 model saved")

# === STEP 4: Generate Grad-CAM based YOLO Dataset ===
gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
classes = sorted(os.listdir(base_path))
out_dir = "/content/yolo_grad"
os.makedirs(f"{out_dir}/images/train", exist_ok=True)
os.makedirs(f"{out_dir}/labels/train", exist_ok=True)

def preprocess_img(img_path):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0), img

def get_bounding_box(cam):
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam_norm = cv2.normalize(cam_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(cam_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if w < 10 or h < 10:
        return None
    return (x, y, w, h)

print("\n📦 Generating YOLOv8 dataset from Grad-CAM...")
for cls in classes:
    cls_dir = os.path.join(base_path, cls)
    for fname in os.listdir(cls_dir):
        try:
            img_path = os.path.join(cls_dir, fname)
            x, original = preprocess_img(img_path)
            preds = model.predict(x)
            pred_class = np.argmax(preds[0])
            cam = gradcam(CategoricalScore(pred_class), x)[0]
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            bbox = get_bounding_box(cam)
            if bbox is None:
                continue
            x_, y_, w_, h_ = bbox
            xc, yc = (x_ + w_ / 2) / IMG_SIZE, (y_ + h_ / 2) / IMG_SIZE
            wn, hn = w_ / IMG_SIZE, h_ / IMG_SIZE
            dst_img_path = os.path.join(out_dir, "images/train", f"{cls}_{fname}")
            shutil.copy(img_path, dst_img_path)
            label_path = dst_img_path.replace("images", "labels").rsplit('.', 1)[0] + ".txt"
            with open(label_path, "w") as f:
                f.write(f"{classes.index(cls)} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
        except Exception as e:
            print(f"⚠️ {fname} failed: {e}")

with open(f"{out_dir}/data.yaml", "w") as f:
    f.write(f"train: {out_dir}/images/train\n")
    f.write(f"val: {out_dir}/images/train\n")
    f.write(f"nc: {len(classes)}\n")
    f.write(f"names: {classes}\n")

print("✅ YOLO dataset ready")

# === STEP 5: Train YOLOv8 ===
print("\n🚀 Training YOLOv8...")
model_yolo = YOLO("yolov8m.pt")
model_yolo.train(data=f"{out_dir}/data.yaml", epochs=50, imgsz=512, batch=8, name="skin_yolov8_v2", patience=10)
print("✅ YOLOv8 training complete")

# === STEP 6: Upload Image & Run Inference ===
print("\n⬆️ Upload a test image")
from google.colab import files
uploaded = files.upload()
test_image_path = list(uploaded.keys())[0]
print(f"✅ Image uploaded: {test_image_path}")

results = model_yolo(test_image_path)
pred = results[0]

if pred.boxes is not None and len(pred.boxes) > 0:
    detected_disease = model_yolo.names[int(pred.boxes.cls[0].item())]
else:
    detected_disease = "Unknown"

print(f"\n🎯 Detected Disease: {detected_disease}")
results.save()
latest_dir = max(glob.glob('runs/detect/exp*'), key=os.path.getmtime)
img = Image.open(os.path.join(latest_dir, os.path.basename(test_image_path)))
plt.imshow(img)
plt.axis('off')
plt.title(f"YOLOv8 Detection: {detected_disease}")
plt.show()

# === STEP 7: Query NVIDIA Gemma ===
print("\n🧠 Querying Gemma model for treatment recommendations...")

headers = {
    "Authorization": "Bearer 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",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

prompt_text = f"""
Detected Disease: {detected_disease}
Please provide:
1. Causes
2. Recommended drugs
3. Natural treatments
4. Specific diet changes
5. Routine changes
6. Best skincare products with online purchase links
"""

payload = {
    "model": "google/gemma-3-27b-it",
    "messages": [{"role": "user", "content": prompt_text}],
    "max_tokens": 512,
    "temperature": 0.2,
    "top_p": 0.7
}

response = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers=headers, json=payload)

if response.status_code == 200:
    reply = response.json()["choices"][0]["message"]["content"]
    print("\n💡 Gemma's Recommendations:\n")
    print(reply)
else:
    print(f"❌ Failed to get response from Gemma (status: {response.status_code})")
    print(response.text)
