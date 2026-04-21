import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class DeepKPTModel(nn.Module):
    def __init__(self, input_dim):
        super(DeepKPTModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)

def train():
    file_path = 'data/OmicsExpressionProteinCodingGenesTPMLogp1.csv'
    if not os.path.exists(file_path):
        print(f"❌ 데이터 파일이 없습니다: {file_path}")
        return

    print("🚀 데이터 로드 중...")
    # 필요 시 nrows=500 유지(빠른 테스트용). 실제 학습 시 제거 가능
    df = pd.read_csv(file_path, index_col=0, nrows=500)

    # 2000개 유전자 선정 및 이름 저장
    selected_genes = df.columns[:2000].tolist()
    data = df[selected_genes].copy()  # (n_samples, 2000)

    os.makedirs('models', exist_ok=True)
    with open('models/gene_list.txt', 'w') as f:
        for gene in selected_genes:
            f.write(f"{gene}\n")

    # ✅ (중요) 학습/추론 전처리 일관성: log2 + (mean/std 저장) + 표준화
    X_np = data.values.astype(np.float32)
    X_np = np.log2(X_np + 1.0)  # inference와 동일

    mean = X_np.mean(axis=0)  # (2000,)
    std = X_np.std(axis=0)    # (2000,)
    std = np.where(std == 0, 1.0, std)

    np.save('models/scaler_mean.npy', mean)
    np.save('models/scaler_std.npy', std)

    X_np = (X_np - mean) / std

    # 라벨링(현재는 임시 라벨) — 기존 흐름 유지
    target_values = X_np.mean(axis=1)
    y_np = (target_values > np.median(target_values)).astype(np.float32)

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = DeepKPTModel(input_dim=2000)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print("🔥 학습 시작...")
    for epoch in range(10):
        epoch_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx).view(-1)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        print(f"Epoch {epoch+1}/10 | loss={epoch_loss/len(loader):.4f}")

    torch.save(model.state_dict(), 'models/best_model.pt')
    print("✅ 학습 완료!")
    print(" - models/gene_list.txt")
    print(" - models/best_model.pt")
    print(" - models/scaler_mean.npy / models/scaler_std.npy")

if __name__ == "__main__":
    train()
