# 🚀 Spaceship Titanic — Variational Quantum Classifier (VQC)

<div align="center">

<!-- Tech Stack Icons -->
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PennyLane](https://img.shields.io/badge/PennyLane-FF6B6B?style=for-the-badge&logo=quantum&logoColor=white)
![JAX](https://img.shields.io/badge/JAX-A020F0?style=for-the-badge&logo=google&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Optax](https://img.shields.io/badge/Optax-FF6F00?style=for-the-badge&logo=google&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Spaceship%20Titanic-20BEFF?style=flat-square&logo=kaggle)](https://www.kaggle.com/competitions/spaceship-titanic/overview)
![Score](https://img.shields.io/badge/Score-0.7220-brightgreen?style=flat-square)
![Approach](https://img.shields.io/badge/Approach-Quantum%20ML-blueviolet?style=flat-square)

</div>

---

## 🇹🇷 Türkçe

### 📖 Proje Hakkında

Bu proje, [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview) yarışması için geliştirilmiş bir **Varyasyonel Kuantum Sınıflandırıcı (VQC)** çözümüdür. Yolcuların uzayda bir anomali sırasında başka bir boyuta ışınlanıp ışınlanmadığını (`Transported`) tahmin etmeyi amaçlamaktadır.

> **Yarışma Puanı: 0.7220** 🎯

### ⚛️ Kuantum Makine Öğrenimi Yaklaşımı

Klasik derin öğrenme yerine **PennyLane** kütüphanesi kullanılarak kuantum devre tabanlı bir model oluşturulmuştur. Temel bileşenler:

| Bileşen | Açıklama |
|---|---|
| **AmplitudeEmbedding** | Özellik vektörünü kuantum durumuna kodlar |
| **StronglyEntanglingLayers** | Parametreli dolaşık kuantum katmanları (n=10) |
| **PauliZ Ölçümü** | Tüm qubit'lerin Z beklenti değerlerinin toplamı |
| **JAX vmap** | Verimliliği artırmak için vektörleştirilmiş işlem |

### 📁 Proje Yapısı

```
SpaceShip Titanic/
├── pennylane_vqc.py        # Ana VQC modeli ve eğitim scripti
├── train.csv               # Eğitim verisi (Kaggle'dan)
├── test.csv                # Test verisi (Kaggle'dan)
├── model_weights_Xlayer.npz  # Kaydedilen model ağırlıkları
└── submission_Xlayer.csv   # Kaggle için gönderim dosyası
```

### 🔧 Kurulum

```bash
# Gerekli kütüphaneleri yükle
pip install pennylane pennylane-lightning jax jaxlib optax optax pandas scikit-learn tqdm numpy
```

### ▶️ Kullanım

Kaggle'dan `train.csv` ve `test.csv` dosyalarını indirip proje klasörüne koyun:

```bash
python pennylane_vqc.py
```

Script çalıştırıldığında:
1. Veriyi yükler ve ön işler
2. Kuantum devreyi kurar
3. 30 epoch boyunca modeli eğitir
4. Model ağırlıklarını `.npz` dosyasına kaydeder
5. `submission.csv` tahmin dosyasını oluşturur

### 🧪 Veri Ön İşleme Adımları

- **PassengerId** ayrıştırma → `GroupId` ve `GroupMemberId`
- **HomePlanet** grup bazlı eksik değer doldurma
- **Cabin** sütununu `CabinDeck`, `CabinNum`, `CabinSide`'a ayırma
- Kategorik değişkenler için `LabelEncoder`
- `MinMaxScaler` ile `[0, π]` aralığına normalleştirme (kuantum devre için)

### ⚙️ Model Hiperparametreleri

| Hiperparametre | Değer |
|---|---|
| Qubit Sayısı | `⌈log₂(n_features)⌉` |
| Kuantum Katman Sayısı | 5 |
| Optimizasyon Algoritması | AdamW |
| Öğrenme Oranı | 0.05 |
| Weight Decay | 0.05 |
| Epoch Sayısı | 30 |

---

## 🇬🇧 English

### 📖 About the Project

This project is a **Variational Quantum Classifier (VQC)** solution developed for the [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview) competition. The goal is to predict whether passengers were `Transported` to an alternate dimension during a spacetime anomaly.

> **Competition Score: 0.7220** 🎯

### ⚛️ Quantum Machine Learning Approach

Instead of classical deep learning, a quantum circuit-based model is built using the **PennyLane** library. Key components:

| Component | Description |
|---|---|
| **AmplitudeEmbedding** | Encodes feature vectors into quantum states |
| **StronglyEntanglingLayers** | Parameterized entangling quantum layers (n=10) |
| **PauliZ Measurement** | Sum of Z-expectation values across all qubits |
| **JAX vmap** | Vectorized batch processing for efficiency |

### 📁 Project Structure

```
SpaceShip Titanic/
├── pennylane_vqc.py        # Main VQC model and training script
├── train.csv               # Training data (from Kaggle)
├── test.csv                # Test data (from Kaggle)
├── model_weights_Xlayer.npz  # Saved model weights
└── submission_Xlayer.csv   # Submission file for Kaggle
```

### 🔧 Installation

```bash
pip install pennylane pennylane-lightning jax jaxlib optax pandas scikit-learn tqdm numpy
```

### ▶️ Usage

Download `train.csv` and `test.csv` from Kaggle and place them in the project folder:

```bash
python pennylane_vqc.py
```

When executed, the script will:
1. Load and preprocess the data
2. Build the quantum circuit
3. Train the model for 30 epochs
4. Save model weights to a `.npz` file
5. Generate a `submission.csv` prediction file

### 🧪 Data Preprocessing Steps

- **PassengerId** parsing → `GroupId` and `GroupMemberId`
- **HomePlanet** group-based missing value imputation
- **Cabin** column split into `CabinDeck`, `CabinNum`, `CabinSide`
- `LabelEncoder` for categorical variables
- `MinMaxScaler` normalization to `[0, π]` range (for quantum encoding)

### ⚙️ Model Hyperparameters

| Hyperparameter | Value |
|---|---|
| Number of Qubits | `⌈log₂(n_features)⌉` |
| Quantum Layers | 5 |
| Optimizer | AdamW |
| Learning Rate | 0.05 |
| Weight Decay | 0.05 |
| Epochs | 30 |

### 📊 Pipeline Overview

```
Raw CSV Data
    │
    ▼
Feature Engineering (GroupId, Cabin split, HomePlanet fill)
    │
    ▼
Encoding + MinMaxScaler [0, π]
    │
    ▼
AmplitudeEmbedding → Quantum Circuit (StronglyEntanglingLayers)
    │
    ▼
PauliZ Expectation Values (+ bias)
    │
    ▼
Binary Cross-Entropy Loss (AdamW optimizer via JAX/Optax)
    │
    ▼
Prediction → submission.csv
```

---

<div align="center">

Made with ❤️ using Quantum Computing · [Kaggle Competition](https://www.kaggle.com/competitions/spaceship-titanic/overview)

</div>
