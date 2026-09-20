# Facial Expression Recognition (RAF-DB)

A modular PyTorch implementation of a custom 4-block Convolutional Neural Network (CNN) for Facial Expression Recognition on the RAF-DB dataset.

---

## 📁 Project Structure

```text
Facial-Expresssion-Recognition/
├── dataset.py        # Phase 1 & 2: Data loading (kagglehub), preprocessing, and augmentations
├── model.py          # Phase 3: Custom 4-block CNN (7x7x512 feature map + 1-layer classifier)
├── train.py          # Phase 4: Training, validation loops, and Early Stopping
├── main.py           # Entry point: CLI argument parsing and pipeline orchestration
├── requirements.txt  # Project dependencies
└── README.md         # Instructions and documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nilkamal21/Facial-Expresssion-Recognition.git
   cd Facial-Expresssion-Recognition
   ```

2. **(Optional) Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

You can run `main.py` directly with default settings, or pass any combination of the 5 tunable hyperparameters in **any order**. Any parameter you omit automatically uses its default value.

### 1. Run with all defaults:
* Defaults: `epochs=30`, `batch_size=64`, `lr=0.001`, `dropout=0.25`, `optimizer=adam`
```bash
python main.py
```

### 2. Change only 2 parameters (in any order):
* E.g., change only `lr` and `epochs`:
```bash
python main.py --lr 0.0005 --epochs 15
```
* Or in reverse order (order does not matter):
```bash
python main.py --epochs 15 --lr 0.0005
```

### 3. Change batch size and optimizer:
```bash
python main.py --batch_size 32 --optimizer sgd
```

### 4. Full custom hyperparameter run:
```bash
python main.py --epochs 40 --batch_size 32 --lr 0.0003 --dropout 0.3 --optimizer adam
```

---

## 🧠 Model Architecture Highlights

* **Input Size**: `[Batch, 3, 112, 112]`
* **4 Convolutional Blocks**:
  * Each block: 2 × `Conv2d` (3x3), 1 × `BatchNorm2d`, 1 × `ReLU`, 1 × `Dropout2d`, 1 × `MaxPool2d` (2x2)
* **Feature Map Output**: Exactly **`[Batch, 512, 7, 7]`**
* **Classifier**: 1 Linear Layer: `nn.Linear(512 * 7 * 7, 7)` (25,088 $\to$ 7 emotion classes)
* **Early Stopping & Smart Checkpointing**: Halts training if validation loss stops improving for 5 consecutive epochs. Saves the best weights with a hyperparameter-specific filename (e.g., `best_model_epochs30_bs64_lr0.001_dropout0.25_adam.pth`) so previous experiments are never overwritten.