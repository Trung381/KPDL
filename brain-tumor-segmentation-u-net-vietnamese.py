"""
Phân đoạn khối u não với U-net
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
import tensorflow.keras.backend as K
import warnings
warnings.filterwarnings("ignore")

"""
Bộ dữ liệu này bao gồm 4237 hình ảnh IRM của các khối u não, mỗi hình ảnh đi kèm với một mặt nạ thủ công được tạo bởi các chuyên gia, 
cho phép xác định chính xác đường viền của các khối u. Bộ dữ liệu này đóng vai trò là tham chiếu để huấn luyện mô hình phân đoạn, 
tập trung vào bốn loại khối u: (glioma, meningioma, pituitary, and no tumor).

Dataset source: https://www.sciencedirect.com/science/article/pii/S095741742302849X#tbl3

Image:
* 0 (No Tumor, 1595 images),
* 1 (Glioma, 649 images),
* 2 (Meningioma, 999 images),
* 3 (Pituitary, 994 images),

Mask:
* 0 (No Tumor, 1595 images),
* 1 (Glioma, 650 images),
* 2 (Meningioma, 1000 images),
* 3 (Pituitary, 994 images),
"""

# Đường dẫn truy cập dữ liệu
data_path = "Brain Tumor Segmentation Dataset"
classes = ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']
images = []
masks = []
labels = []
target_size = (128, 128)

# Tải hình ảnh và mặt nạ
for class_name in classes:
    image_folder = os.path.join(data_path, 'image', str(classes.index(class_name)))
    mask_folder = os.path.join(data_path, 'mask', str(classes.index(class_name)))
    
    if os.path.exists(image_folder) and os.path.exists(mask_folder):
        for image_name in tqdm(os.listdir(image_folder), desc=class_name):
            if image_name.endswith('.jpg') or image_name.endswith('.png'):
                image_path = os.path.join(image_folder, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, target_size)
                image = image / 255.0
                
                mask_name = image_name.replace('.jpg', '_m.jpg').replace('.png', '_m.png')
                mask_path = os.path.join(mask_folder, mask_name)
                
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, target_size)
                    mask = mask / 255.0
                    
                    images.append(image)
                    masks.append(mask)
                    labels.append(classes.index(class_name))
                else:
                    print(f"Mặt nạ bị thiếu cho hình ảnh: {image_name}")

# Chuyển đổi danh sách thành mảng NumPy
images = np.array(images).reshape(-1, 128, 128, 1)
masks = np.array(masks).reshape(-1, 128, 128, 1)
labels = np.array(labels)

print(f"Số lượng hình ảnh: {len(images)}, Số lượng mặt nạ: {len(masks)}, Số lượng nhãn: {len(labels)}")

# Trực quan hóa dữ liệu
unique, counts = np.unique(labels, return_counts=True)
class_counts = dict(zip(classes, counts))
total_images = sum(class_counts.values())
percentages = [(count / total_images) * 100 for count in class_counts.values()]

plt.figure(figsize=(8, 6))
bars = plt.bar(class_counts.keys(), class_counts.values(), color='#E17A8A')
for bar, percentage in zip(bars, percentages):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{percentage:.1f}%', ha='center', va='bottom')
plt.xlabel('Các lớp khối u')
plt.ylabel("Số lượng hình ảnh")
plt.title("Số lượng hình ảnh theo lớp khối u não")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

"""
Các lớp thể hiện sự phân bố không cân bằng, với lớp "no tumor" chiếm tỷ lệ đáng kể (37.6%), 
trong khi các loại khối u khác có tỷ lệ thấp hơn. Sự mất cân bằng này có thể ảnh hưởng đến hiệu suất của mô hình U-Net
"""

# Trực quan hóa các ví dụ
fig, axs = plt.subplots(len(classes) + 1, 5, figsize=(10, 10), 
                       gridspec_kw={'width_ratios': [0.5, 4, 4, 4, 4], 'height_ratios': [0.5, 4, 4, 4, 4]})
axs[0, 0].text(0.5, 0.5, "", fontsize=10, ha='center', va='center', fontweight='bold')
axs[0, 0].axis('off')
axs[0, 1].text(0.5, 0.5, "Ví dụ 1", fontsize=10, ha='center', va='center', fontweight='bold')
axs[0, 1].axis('off')
axs[0, 2].text(0.5, 0.5, "Mặt nạ 1", fontsize=10, ha='center', va='center', fontweight='bold')
axs[0, 2].axis('off')
axs[0, 3].text(0.5, 0.5, "Ví dụ 2", fontsize=10, ha='center', va='center', fontweight='bold')
axs[0, 3].axis('off')
axs[0, 4].text(0.5, 0.5, "Mặt nạ 2", fontsize=10, ha='center', va='center', fontweight='bold')
axs[0, 4].axis('off')

for i, class_name in enumerate(classes):
    class_index = classes.index(class_name)
    example_index1 = np.where(labels == class_index)[0][0]
    example_index2 = np.where(labels == class_index)[0][2]
    image1 = images[example_index1]
    mask1 = masks[example_index1]
    image2 = images[example_index2]
    mask2 = masks[example_index2]
    
    axs[i + 1, 1].imshow(image1, cmap='gray')
    axs[i + 1, 1].axis('off')
    axs[i + 1, 2].imshow(mask1, cmap='gray')
    axs[i + 1, 2].axis('off')
    axs[i + 1, 3].imshow(image2, cmap='gray')
    axs[i + 1, 3].axis('off')
    axs[i + 1, 4].imshow(mask2, cmap='gray')
    axs[i + 1, 4].axis('off')
    axs[i + 1, 0].text(0.5, 0.5, class_name, fontsize=10, ha='center', va='center', fontweight='bold')
    axs[i + 1, 0].axis('off')

plt.tight_layout()
plt.show()

# Chuẩn bị dữ liệu
images_train, images_val, masks_train, masks_val, labels_train, labels_val = train_test_split(
    images, masks, labels, test_size=0.3, random_state=42, stratify=labels, shuffle=True
)

print(f"Bộ dữ liệu huấn luyện - Hình ảnh: {images_train.shape}, Mặt nạ: {masks_train.shape}, Nhãn: {labels_train.shape}")
print(f"Bộ dữ liệu xác thực - Hình ảnh: {images_val.shape}, Mặt nạ: {masks_val.shape}, Nhãn: {labels_val.shape}")

def train_generator(images, masks, batch_size, seed=42):
    image_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    mask_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    image_generator = image_datagen.flow(images, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(masks, batch_size=batch_size, seed=seed)
    
    while True:
        img_batch = next(image_generator)
        mask_batch = next(mask_generator)
        yield img_batch, mask_batch

def plot_images(images, masks, num_images=5):
    plt.figure(figsize=(15, 6))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i].reshape(128, 128), cmap='gray')
        plt.axis('off')
        plt.title('Hình ảnh')
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(masks[i].reshape(128, 128), cmap='gray')
        plt.axis('off')
        plt.title('Mặt nạ')
    plt.tight_layout()
    plt.show()

images_batch, masks_batch = next(train_generator(images_train, masks_train, batch_size=5))
plot_images(images_batch, masks_batch, num_images=5)

"""
Các chỉ số **Dice Coefficient (DICE), Dice Loss và Intersection over Union (IoU)** thường được sử dụng để đánh giá hiệu suất của các mô hình phân đoạn hình ảnh, đặc biệt trong các nhiệm vụ phân đoạn y tế:
"""

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def iou(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    total = K.sum(y_true_f) + K.sum(y_pred_f)
    union = total - intersection
    return (intersection + smooth) / (union + smooth)

def unet_model(input_shape):
    inputs = layers.Input(input_shape)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Dropout(0.2)(c4)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Dropout(0.3)(c5)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Dropout(0.2)(c7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Dropout(0.1)(c9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

input_shape = (128, 128, 1)
model = unet_model(input_shape)
model.summary()

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', dice_coefficient, iou])

model_checkpoint = ModelCheckpoint('best_unetmodel.keras', monitor='val_dice_coefficient', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_dice_coefficient', patience=10, mode='max', verbose=1)

batch_size = 32
train_gen = train_generator(images_train, masks_train, batch_size=batch_size)

history = model.fit(
    train_gen,
    steps_per_epoch=len(images_train) // batch_size,
    epochs=100,
    validation_data=(images_val, masks_val),
    callbacks=[model_checkpoint, early_stopping]
)

# Đánh giá
train_loss, train_accuracy, train_dice_coef, train_iou_coef = model.evaluate(images_train, masks_train, verbose=1)
print(f'Mất mát huấn luyện: {train_loss:.4f}')
print(f'Độ chính xác huấn luyện: {train_accuracy:.4f}')
print(f'Hệ số Dice huấn luyện: {train_dice_coef:.4f}')
print(f'Hệ số IoU huấn luyện: {train_iou_coef:.4f}')

val_loss, val_accuracy, val_dice_coef, val_iou_coef = model.evaluate(images_val, masks_val, verbose=1)
print(f'Mất mát xác thực: {val_loss:.4f}')
print(f'Độ chính xác xác thực: {val_accuracy:.4f}')
print(f'Hệ số Dice xác thực: {val_dice_coef:.4f}')
print(f'Hệ số IoU xác thực: {val_iou_coef:.4f}')

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs[0, 0].plot(history.history['loss'], label='Mất mát huấn luyện')
axs[0, 0].plot(history.history['val_loss'], label='Mất mát xác thực')
axs[0, 0].set_title('Mất mát qua các Epoch')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Mất mát')
axs[0, 0].set_ylim(0, 1)
axs[0, 0].legend()
axs[0, 1].plot(history.history['accuracy'], label='Độ chính xác huấn luyện')
axs[0, 1].plot(history.history['val_accuracy'], label='Độ chính xác xác thực')
axs[0, 1].set_title('Độ chính xác qua các Epoch')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Độ chính xác')
axs[0, 1].set_ylim(0, 1)
axs[0, 1].legend()
axs[1, 0].plot(history.history['dice_coefficient'], label='Hệ số Dice huấn luyện')
axs[1, 0].plot(history.history['val_dice_coefficient'], label='Hệ số Dice xác thực')
axs[1, 0].set_title('Hệ số Dice qua các Epoch')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Hệ số Dice')
axs[1, 0].legend()
axs[1, 1].plot(history.history['iou'], label='IoU huấn luyện')
axs[1, 1].plot(history.history['val_iou'], label='IoU xác thực')
axs[1, 1].set_title('IoU qua các Epoch')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('IoU')
axs[1, 1].legend()
plt.tight_layout()
plt.show()

predictions = model.predict(images_val)
predictions = (predictions > 0.5).astype(np.uint8)

n = 5
plt.figure(figsize=(15, 15))
for i in range(n):
    plt.subplot(n, 3, i * 3 + 1)
    plt.imshow(images_val[i+1].reshape(128, 128), cmap='gray')
    plt.title('Hình ảnh đầu vào', fontsize=10)
    plt.axis('off')
    plt.subplot(n, 3, i * 3 + 2)
    plt.imshow(masks_val[i+1].reshape(128, 128), cmap='gray')
    plt.title('Mặt nạ thực tế', fontsize=10)
    plt.axis('off')
    plt.subplot(n, 3, i * 3 + 3)
    plt.imshow(predictions[i+1].reshape(128, 128), cmap='gray')
    plt.title('Mặt nạ dự đoán', fontsize=10)
    plt.axis('off')
plt.tight_layout()
plt.show()

# Dự đoán trên một bộ dữ liệu khác
test_dir = '/kaggle/input/brain-tumor-classification-mri/Training'
test_images = []
test_labels = []

for class_name in classes:
    class_folder = os.path.join(test_dir, class_name)
    image_names = os.listdir(class_folder)
    if len(image_names) >= 2:
        for image_name in image_names[:2]:
            image_path = os.path.join(class_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, target_size)
            image = image / 255.0
            image = np.expand_dims(image, axis=-1)
            test_images.append(image)
            test_labels.append(class_name)

test_images = np.array(test_images)
test_predictions = model.predict(test_images)
test_predictions = (test_predictions > 0.5).astype(np.uint8)

fig, axs = plt.subplots(len(classes) + 1, 5, figsize=(15, 10), 
                       gridspec_kw={'width_ratios': [0.5, 4, 4, 4, 4], 'height_ratios': [0.5] + [4]*len(classes)})
titles = ["Lớp", "Hình ảnh 1", "Mặt nạ dự đoán 1", "Hình ảnh 2", "Mặt nạ dự đoán 2"]
for j in range(5):
    axs[0, j].text(0.5, 0.5, titles[j], fontsize=10, ha='center', va='center', fontweight='bold')
    axs[0, j].axis('off')

for i, class_name in enumerate(classes):
    class_indices = [j for j, label in enumerate(test_labels) if label == class_name]
    if len(class_indices) >= 2:
        example_index1 = class_indices[0]
        example_index2 = class_indices[1]
        axs[i + 1, 0].text(0.5, 0.5, class_name, fontsize=10, ha='center', va='center', fontweight='bold')
        axs[i + 1, 0].axis('off')
        axs[i + 1, 1].imshow(test_images[example_index1].squeeze(), cmap='gray')
        axs[i + 1, 1].axis('off')
        axs[i + 1, 2].imshow(test_predictions[example_index1], cmap='gray')
        axs[i + 1, 2].axis('off')
        axs[i + 1, 3].imshow(test_images[example_index2].squeeze(), cmap='gray')
        axs[i + 1, 3].axis('off')
        axs[i + 1, 4].imshow(test_predictions[example_index2], cmap='gray')
        axs[i + 1, 4].axis('off')

plt.tight_layout()
plt.show()