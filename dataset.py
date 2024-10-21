import numpy as np
import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import hashlib

# CIFAR-100 데이터셋 로드
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_data = CIFAR100(root="./data/CIFAR", train=True, download=True, transform=transform)
test_data = CIFAR100(root="./data/CIFAR", train=False, download=True, transform=transform)

# CIFAR-100의 전체 이미지와 라벨
cifar_images = np.concatenate([train_data.data, test_data.data], axis=0)  # (60000, 32, 32, 3)
cifar_labels = np.concatenate([train_data.targets, test_data.targets], axis=0)  # (60000,)

# testset.npy 데이터 로드
x_test = np.load('data/testset.npy')  # (N, 32, 32, 3) 형태

# testset.npy의 각 이미지에 대해 CIFAR-100에서 가장 가까운 이미지의 라벨을 찾기
matched_labels = []
# 해시 생성 함수 (MD5 사용)
def image_to_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()

# CIFAR-100 이미지에 대해 해시값 생성
cifar_image_hashes = {image_to_hash(image): label for image, label in zip(cifar_images, cifar_labels)}

# testset.npy의 각 이미지에 대해 CIFAR-100에서 동일한 이미지의 라벨을 찾기
matched_labels = []

# 해시값을 비교하여 일치하는 CIFAR-100 이미지의 라벨 찾기
for test_image in tqdm(x_test, desc="Matching images using hash"):
    test_hash = image_to_hash(test_image)
    matched_label = cifar_image_hashes.get(test_hash, None)  # 동일한 이미지의 라벨을 가져옴
    matched_labels.append(matched_label)

np.save('data/testlabel.npy', np.array(matched_labels))