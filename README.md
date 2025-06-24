# NUmPYTermAi

A lightweight LLM training script for Termux built with **NumPy**, **math**, **os**, and **pickle**.

> ✅ Train simple language models directly on your Android device  
> ✅ Fully self-contained — no TensorFlow/PyTorch needed  
> ✅ Powered by NumPy for fast, efficient matrix operations  
> ✅ Easy to understand and hack on!

---

## 📂 Contents

| File                  | Description                                |
|------------------------|--------------------------------------------|
| `train_generalize.py`  | Main training script for generalizing model |
| `train_memorize.py`    | Training script focused on memorization    |
| `dataset.txt`          | Core dataset used for training             |
| `example.dataset.txt`  | Example input data for quick testing      |
| `Infer`                | CLI inference utility                     |

---

## 🛠️ Requirements

- Python 3.x
- NumPy
- (Optional) Other standard libraries (`math`, `os`, `pickle`)

You can install NumPy via:
```bash
pip install numpy


---

🚀 Usage

1. Train

Run the training scripts as follows:

python3 train_generalize.py

Or:

python3 train_memorize.py

2. Run Inference

After training:

python3 Infer


---

💡 Notes

Designed to work in Termux on Android.

Training can take some time depending on dataset size.

---

📜 License

This project is licensed under the MIT License.


---

Happy hacking! 🎉
