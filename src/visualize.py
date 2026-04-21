import matplotlib.pyplot as plt
import pandas as pd

def plot_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss', color='#2E7D32', linewidth=2)
    plt.title('Deep-KPT Model: Training Loss Convergence', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (Binary Cross Entropy)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('models/loss_curve.png') # 이미지로 저장
    plt.show()

# 만약 train.py에서 loss_history를 리스트로 담아두셨다면 이 함수를 호출하면 됩니다.