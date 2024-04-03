# 모델설계 작성
# 모델은 하나의 입력층과 각각 배치정규화, 패딩 및 ReLU 활성화가 있는
# 8개의  Conv 계층, 5개의 Pool 계층 (4개의 Conv-Pool-Conv 그룹), 1개의 드롭아웃 계층, 2개의 완전연결(FC) 계층 및 1개의 출력층을 갖는 2D CNN 모델
# 첫 번째 Conv층은 크기(224×224)의 입력 학습 웨이퍼 이미지에서 특징을 추출
# 각 Conv층에는 고유한 피처맵을 추출하기 위한 학습 가능한 필터 세트가 포함,  Conv층의 깊이가 증가 함에 따라 필터의 수가 증가하므로 피처맵의 수도 증가
# 첫 번째, 두 번째, 세 번째, 네 번째 Conv-Pool-Conv 그룹에 대해 각각 16, 32, 64, 128개의 피처맵을 사용
# 각 Conv 및 Pool 계층은 각각 크기가 3x3 및 2x2인 서브샘플링 필터로 구성
# ReLU 활성화 함수는 Pool층과 출력층을 제외한 모든 계층에 적용
# CNN 모델의 전체 학습 과정에서 VGP를 해결하기 위해 BN 연산을 사용
# Conv층과 마지막 풀링층 사이에 SD 함수를 0.2의 비율로 적용
# 입력 및 출력 피처맵의 차원을 동일하게 유지하기 위해 모든 Conv층에는 제로 패딩(padding)이 적용
# Softmax 활성화 함수가 모델의 출력층에 적용
# 모멘텀 최적화와 RMSProp(Root Mean Squared Prop)의 개념을 결합한 Adam 최적화 방법을 옵티마이저로 선택
# 배치 크기와 에포크(epoch) 수와 같은 몇몇 다른 매개변수에 각각 100 및 20이 할당
import torch
import torch.nn as nn


class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()

        # 첫 번째 Conv층
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        # 두 번째 Conv층
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        # 세 번째 Conv층
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        # 네 번째 Conv층
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        # 다섯 번째 Conv층
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        # 여섯 번째 Conv층
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)

        # 일곱 번째 Conv층
        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)

        # 여덟 번째 Conv층
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(128)

        # 풀링층
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 드롭아웃
        self.dropout = nn.Dropout(0.5)

        # 완전 연결(FC) 계층
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)  # 출력층

    def forward(self, x):
        # 첫 번째 Conv층과 ReLU 활성화 함수
        x = self.relu(self.bn1(self.conv1(x)))

        # 두 번째 Conv층과 ReLU 활성화 함수
        x = self.relu(self.bn2(self.conv2(x)))

        # 첫 번째 풀링층
        x = self.pool(x)

        # 세 번째 Conv층과 ReLU 활성화 함수
        x = self.relu(self.bn3(self.conv3(x)))

        # 네 번째 Conv층과 ReLU 활성화 함수
        x = self.relu(self.bn4(self.conv4(x)))

        # 두 번째 풀링층
        x = self.pool(x)

        # 다섯 번째 Conv층과 ReLU 활성화 함수
        x = self.relu(self.bn5(self.conv5(x)))

        # 여섯 번째 Conv층과 ReLU 활성화 함수
        x = self.relu(self.bn6(self.conv6(x)))

        # 세 번째 풀링층
        x = self.pool(x)

        # 일곱 번째 Conv층과 ReLU 활성화 함수
        x = self.relu(self.bn7(self.conv7(x)))

        # 여덟 번째 Conv층과 ReLU 활성화 함수
        x = self.relu(self.bn8(self.conv8(x)))

        # 네 번째 풀링층
        x = self.pool(x)

        # 평탄화
        x = x.view(-1, 128 * 7 * 7)

        # FC 계층과 ReLU 활성화 함수
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# 모델 인스턴스 생성
model = DeepCNN()

# 모멘텀 최적화와 RMSProp(Root Mean Squared Prop)의 개념을 결합한 Adam 최적화 방법을 옵티마이저로 선택
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

# 배치 크기와 에포크(epoch) 수와 같은 몇몇 다른 매개변수에 각각 100 및 20이 할당
batch_size = 100
epochs = 20

