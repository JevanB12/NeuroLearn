# NeuroLearn 🧠  
### Coursework Project — Loughborough University  
### Module: AI Methods (COB107) — Neural Network Implementation

---

## 📘 Overview

**NeuroLearn** is a Python-based implementation of a **Multilayer Perceptron (MLP)** neural network, developed as part of the **AI Methods (COB107)** coursework at **Loughborough University**.  
The project aims to predict **future water level values** based on historical river data from the **Ouse93-96 dataset**, demonstrating the use of **artificial neural networks (ANNs)** for time-series forecasting.  

The model is built **from scratch**, without using any pre-existing deep learning libraries such as TensorFlow or PyTorch, adhering to coursework requirements.

---

## 🎯 Objectives

- Implement a fully functional **feedforward neural network** using Python.  
- Apply **backpropagation** for network training and weight optimization.  
- Introduce **momentum** and **learning rate annealing** to enhance convergence.  
- Perform **data preprocessing**, normalization, and training/test splitting.  
- Evaluate the model’s performance through **loss** and **RMSE** metrics.  

---

## 🧠 Neural Network Architecture

The implemented MLP consists of:
- **Input Layer** — accepts normalized feature vectors from the dataset  
- **Hidden Layer 1:** 10 neurons  
- **Hidden Layer 2:** 8 neurons  
- **Hidden Layer 3:** 6 neurons  
- **Hidden Layer 4:** 4 neurons  
- **Output Layer:** 1 neuron (predicting the next river level value)  

Each neuron uses a **sigmoid activation function**, and training is performed using **momentum-based backpropagation** with **annealed learning rates**.

---

## 🧩 Features

- **Feedforward MLP network built entirely from scratch**  
- **Momentum-based optimization** to reduce oscillations during training  
- **Learning rate annealing** for improved convergence stability  
- **Dynamic plotting** of Loss and RMSE values across training epochs  
- **Full data preprocessing pipeline** using pandas and NumPy  
- **Error handling** for data normalization and cleaning stages  

---

## 🧮 Dataset

**Dataset:** `Ouse93-96 - Student.xlsx`  
This dataset contains **river level and environmental data** recorded from 1993–1996.  
The model was trained to predict **the next time step’s river level** based on previous values.

### Preprocessing Steps:
- Conversion of all columns to numeric types  
- Handling of missing and invalid data entries  
- Removal of NaN rows  
- Standardization using z-score normalization  
- Splitting into input features (X) and target variable (y)  

---

## 🧰 Technologies Used

- **Python 3.10+**  
- **NumPy** — Matrix operations and numerical computations  
- **Pandas** — Data handling and preprocessing  
- **Matplotlib** — Visualization of model training (Loss and RMSE plots)

---

## 🚀 How to Run

1. **Ensure dependencies are installed:**
   ```bash
   pip install numpy pandas matplotlib
   ```

2. **Place the dataset** (`Ouse93-96 - Student.xlsx`) in the project directory.

3. **Run the program:**
   ```bash
   python neurolearn.py
   ```

4. The script will:
   - Load and clean the dataset  
   - Train the MLP model using backpropagation  
   - Print progress every 100 epochs  
   - Display a plot showing **Loss** and **RMSE** trends  

---

## 📊 Key Parameters

| Parameter | Description | Default |
|------------|-------------|----------|
| `learning_rate` | Initial learning rate | 0.01 |
| `momentum` | Weight update momentum | 0.9 |
| `epochs` | Total number of training iterations | 1000 |
| `annealing_rate` | Learning rate decay factor per epoch | 0.99 |

---

## 📈 Evaluation

The model performance is tracked using:
- **Loss (Mean Squared Error)**  
- **RMSE (Root Mean Squared Error)**  

These metrics are plotted over the course of training to evaluate convergence and stability.

---

## 🧩 File Structure

```
NeuroLearn/
│
├── neurolearn.py                     # MLP implementation and training pipeline
├── Ouse93-96 - Student.xlsx          # Dataset (input data)
├── Case study_Application of ANNs.pdf # Coursework specification
└── README.md                         # Project documentation
```

---

## 🧠 Learning Outcomes

Through this project, the following concepts were implemented and demonstrated:
- Design and implementation of **multilayer perceptron architectures**  
- Application of **backpropagation** with **momentum**  
- Use of **learning rate annealing** to improve model performance  
- **Normalization and preprocessing** of real-world data  
- **Visualization and interpretation** of ANN training behavior  

---

## 🏫 Coursework Context

This project was completed as part of the **AI Methods (COB107)** module at **Loughborough University**, under the supervision of **Dr. C. W. Dawson**.  
It accounts for **20% of the module assessment** and demonstrates applied neural network implementation and evaluation skills using Python.

---

## 📄 License

This project is for **academic use only** as part of the **AI Methods coursework at Loughborough University**.  
Any reuse or modification must credit the original author and coursework module.

---

**NeuroLearn — Predictive Neural Network for River Level Forecasting**  
*Developed for the AI Methods (COB107) coursework at Loughborough University.*
