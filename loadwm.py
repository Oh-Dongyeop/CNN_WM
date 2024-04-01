import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

data_folder = 'wm_images'
output_folder = 'processed_data'
output_subfolders = ['train', 'val', 'test']

# 데이터 증강 비율 설정
rotation_angle = 10
flip_chance = 0.5
shift_percentage = 0.2
shear_range = 0.1
channel_shift_range = 0.1

# 데이터 증강 함수 정의
def augment_image(image):
    # 회전
    angle = np.random.uniform(-rotation_angle, rotation_angle)
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

    # 좌우 반전
    if np.random.rand() < flip_chance:
        image = cv2.flip(image, 1)

    # 너비 이동
    shift = int(shift_percentage * cols)
    shift = np.random.randint(-shift, shift)
    image = np.roll(image, shift, axis=1)

    # 높이 이동
    shift = int(shift_percentage * rows)
    shift = np.random.randint(-shift, shift)
    image = np.roll(image, shift, axis=0)

    # 전단 변환
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    shear_matrix = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, shear_matrix, (cols, rows))

    # 채널 이동
    channel_shift_val = np.random.uniform(-channel_shift_range, channel_shift_range)
    image = image.astype(np.float32)
    image += channel_shift_val
    image = np.clip(image, 0., 255.)
    image = image.astype(np.uint8)

    return image

# 데이터 폴더 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for folder in output_subfolders:
    if not os.path.exists(os.path.join(output_folder, folder)):
        os.makedirs(os.path.join(output_folder, folder))

# 클래스별 데이터 불러오기 및 증강
class_images = {}
total_images = 0
for folder in os.listdir(data_folder):
    class_images[folder] = []
    class_folder = os.path.join(data_folder, folder)
    for filename in os.listdir(class_folder):
        image_path = os.path.join(class_folder, filename)
        image = cv2.imread(image_path)
        class_images[folder].append(image)
        total_images += 1

# 각 클래스당 최소 10,000개의 이미지를 유지하면서 증강
min_images_per_class = 10000
processed_images = []
for folder, images in class_images.items():
    if len(images) >= min_images_per_class:
        selected_indices = np.random.choice(len(images), min_images_per_class, replace=False)
        selected_images = [images[i] for i in selected_indices]
    else:
        selected_images = images.copy()
        while len(selected_images) < min_images_per_class:
            selected_indices = np.random.choice(len(images), min_images_per_class - len(selected_images), replace=True)
            selected_images.extend([images[i] for i in selected_indices])
    for image in selected_images:
        processed_images.append((image, folder))

# 데이터 분할
X = [item[0] for item in processed_images]
y = [item[1] for item in processed_images]
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2/0.85, random_state=42)

# 데이터 저장
np.savez(os.path.join(output_folder, 'train.npz'), X=X_train, y=y_train)
np.savez(os.path.join(output_folder, 'val.npz'), X=X_val, y=y_val)
np.savez(os.path.join(output_folder, 'test.npz'), X=X_test, y=y_test)
