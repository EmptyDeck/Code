# 라즈베리파이에 들어갈 코드

from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose
from torchaudio.transforms import MFCC
from torchaudio.datasets import URBANSOUND8K
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.optim as optim
import torch.nn as nn
import os
import torch
import torchaudio
import torchaudio.transforms as T
from gpiozero import MCP3008
from time import sleep
from RPi import GPIO
import numpy as np

# 모델 로드

model = torch.load("path/to/trained_model.pth")
model.eval()

# 마이크를 사용하기 위한 설정

adc = MCP3008(channel=0)

# 소리 종류 리스트

sound_categories = ["car_horn", "bicycle_bell", "siren", "fire_alarm", "doorbell",
                    "baby_crying", "dog_barking", "glass_breaking", "gunshot", "shouting"]

# 소리를 전처리하는 함수


def preprocess_sound(sound):  # 소리 데이터를 파형(waveform)으로 변환


waveform, sample_rate = torchaudio.load(sound, num_frames=16000)

   # 리샘플링
   resampler = T.Resample(orig_freq=sample_rate,
                          new_freq=8000, dtype=torch.float32)
    waveform = resampler(waveform)

    # 멜 스펙트로그램으로 변환
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=8000, n_fft=400, hop_length=200, n_mels=128)
    mel_spec = mel_spectrogram(waveform)

    # 아무리 작은 값이라도 로그를 취할 수 있도록 log-mel 스펙트로그램 계산
    log_mel_spec = T.AmplitudeToDB(top_db=80)(mel_spec)

    # 정규화
    normalize = T.Normalize(mean=log_mel_spec.mean(), std=log_mel_spec.std())
    log_mel_spec_normalized = normalize(log_mel_spec)

    return log_mel_spec_normalized

# 소리를 인식하고 분류하는 함수


def classify*sound(sound):


preprocessed_sound = preprocess_sound(sound)
output = model(preprocessed_sound.unsqueeze(0))
*, predicted = torch.max(output, 1)

   return predicted.item()

# 소리 인식 및 디스플레이 출력

while True:
sound = adc.value
processed_sound = preprocess_sound(sound)
result = classify_sound(processed_sound)

   # 디스플레이에 결과 출력
   print("Detected sound category:", result)
    sleep(1)

---

# 학습 코드


# 데이터셋 클래스 정의


class UrbanSoundDataset(Dataset):
def **init**(self, data, labels, transform=None):


self.data = data
self.labels = labels
self.transform = transform

   def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sound, label = self.data[idx], self.labels[idx]
        if self.transform:
            sound = self.transform(sound)
        return sound, label

# 데이터 로드 및 전처리

urbansound*data = URBANSOUND8K("UrbanSound8K")  # UrbanSound8K 데이터셋을 로드합니다.
sounds, labels = [], []
for sound, label, \** in urbansound_data:  # 데이터셋에서 소리와 레이블을 추출합니다.
sounds.append(sound)
labels.append(label)

train_sounds, val_sounds, train_labels, val_labels = train_test_split(sounds, labels, test_size=0.2, random_state=42)  # 훈련 및 검증 데이터셋으로 나눕니다.

# 특성 추출 파이프라인 정의

mfcc_transform = MFCC(sample_rate=16000, n_mfcc=40, log_mels=True)  # MFCC 특성 변환을 위한 파이프라인을 정의합니다.

# 데이터셋 및 데이터 로더 생성

train_dataset = UrbanSoundDataset(train_sounds, train_labels, transform=mfcc_transform)  # 훈련 데이터셋을 생성합니다.
val_dataset = UrbanSoundDataset(val_sounds, val_labels, transform=mfcc_transform)  # 검증 데이터셋을 생성합니다.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 훈련 데이터 로더를 생성합니다.
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 검증 데이터 로더를 생성합니다.

# 모델 구축



## Batch Normalization (배치 정규화):

# 신경망 학습 도중 각 층의 입력 분포가 변하는 현상인 내부 공변량 변화(Internal Covariate Shift)를 줄이는 방법입니다.
# 배치 정규화는 각 층의 활성화 출력을 정규화하여 학습 속도를 높이고, 과적합을 줄이며, 더 큰 학습률을 사용할 수 있게 합니다.
# Global Average Pooling (전역 평균 풀링):

# 컨볼루션 층의 출력 특성 맵에 대해 전역 평균을 계산하여 공간 차원을 제거하는 방법입니다.
# Global Average Pooling을 사용하면 과적합을 줄일 수 있으며, 모델의 파라미터 수를 줄일 수 있습니다. 이는 Fully Connected 층을 사용하는 것보다 계산 효율성이 높고, 일반화 능력이 더 좋다고 알려져 있습니다.


# AudioClassifier: 주변 소리를 분류하는 딥러닝 모델
class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()

        # 첫 번째 컨볼루션 층 (Convolutional Layer)
        # 입력 채널: 1, 출력 채널: 16, 커널 크기: 3x3, 패딩: 1
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # Batch Normalization 층 (입력 채널: 16)
        self.bn1 = nn.BatchNorm2d(16)
        # 활성화 함수 (ReLU)
        self.relu = nn.ReLU()
        # 최대 풀링 층 (Max Pooling Layer), 커널 크기: 2x2
        self.max_pool1 = nn.MaxPool2d(2, 2)

        # 두 번째 컨볼루션 층
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.max_pool2 = nn.MaxPool2d(2, 2)

        # 세 번째 컨볼루션 층
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.max_pool3 = nn.MaxPool2d(2, 2)

        # Global Average Pooling 층
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully Connected 층 (입력: 64, 출력: num_classes)
        self.fc = nn.Linear(64, num_classes)

        # 드롭아웃 층 (Dropout Layer), 드롭아웃 비율: 0.5
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 첫 번째 컨볼루션 층
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool1(x)

        # 두 번째 컨볼루션 층
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.max_pool2(x)

        # 세 번째 컨볼루션 층
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.max_pool3(x)

        # Global Average Pooling 층
        x = self.global_avg_pool(x)
        x = x.view(-1, 64)

        # 드롭아웃 적용
        x = self.dropout(x)

        # Fully Connected 층을 통해 최종 출력
        x = self.fc(x)
        return x

# 모델 초기화


model = AudioClassifier(num_classes=10).cuda() # AudioClassifier 모델을 초기화하고 GPU로 이동합니다.
criterion = nn.CrossEntropyLoss()  # 손실 함수로 CrossEntropyLoss를 사용합니다.
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 옵티마이저로 Adam을 사용합니다.

# 훈련 및 검증 함수


def train(model, dataloader, criterion, optimizer):

model.train() # 모델을 훈련 모드로 설정합니다.
running*loss = 0.0
running_corrects = 0
for inputs, labels in dataloader:
inputs, labels = inputs.cuda(), labels.cuda()  # 입력 데이터와 레이블을 GPU로 이동합니다.
optimizer.zero_grad()  # 그래디언트를 초기화합니다.
outputs = model(inputs.unsqueeze(1))  # 모델을 통과시켜 예측값을 계산합니다.
*, preds = torch.max(outputs, 1)  # 예측값 중 가장 큰 값을 선택합니다.
loss = criterion(outputs, labels)  # 손실을 계산합니다.
loss.backward()  # 역전파를 수행합니다.
optimizer.step()  # 옵티마이저를 사용하여 모델의 가중치를 업데이트합니다.
running_loss += loss.item() \* inputs.size(0)
running_corrects += torch.sum(preds == labels.data)
return running_loss / len(dataloader.dataset), running_corrects.double() / len(dataloader.dataset)


def validate(model, dataloader, criterion):

model.eval() # 모델을 평가 모드로 설정합니다.
running*loss = 0.0
running_corrects = 0
with torch.no_grad():
for inputs, labels in dataloader:
inputs, labels = inputs.cuda(), labels.cuda()  # 입력 데이터와 레이블을 GPU로 이동합니다.
outputs = model(inputs.unsqueeze(1))  # 모델을 통과시켜 예측값을 계산합니다.
*, preds = torch.max(outputs, 1)  # 예측값 중 가장 큰 값을 선택합니다.
loss = criterion(outputs, labels)  # 손실을 계산합니다.
running_loss += loss.item() \* inputs.size(0)
running_corrects += torch.sum(preds == labels.data)
return running_loss / len(dataloader.dataset), running_corrects.double() / len(dataloader.dataset)

# 훈련 및 검증 루프

num_epochs = 25
for epoch in range(num_epochs):
train_loss, train_acc = train(model, train_loader, criterion, optimizer)  # 훈련 함수를 호출하여 훈련합니다.
val_loss, val_acc = validate(model, val_loader, criterion)  # 검증 함수를 호출하여 검증합니다.
print(f"Epoch {epoch+1}/{num_epochs}: train_loss {train_loss:.4f}, train_acc {train_acc:.4f}, val_loss {val_loss:.4f}, val_acc {val_acc:.4f}")  # 에포크별로 훈련 및 검증 결과를 출력합니다.
