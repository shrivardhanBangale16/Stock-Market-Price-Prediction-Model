# Stock Market Price Prediction using LSTM and Streamlit

A deep learning-powered web application that predicts stock prices using LSTM (Long Short-Term Memory) neural networks. 
It features interactive visualization, real-time forecasting, and a clean user interface built with Streamlit.

---

## 📌 Features
- 📈 Predict future stock prices based on historical trends
- 🧠 Uses LSTM neural networks for time-series forecasting
- 🧩 Includes moving average analysis (MA50, MA100, MA200)
- 🌐 Built with Streamlit for interactive web deployment
- 📊 Supports both Indian and global stock tickers

---

## 📸 Screenshots
> Add your app screenshots in the `assets/` folder and link them here:
```
![App Screenshot](assets/screenshot.png)
```

---

## 🛠️ Tech Stack / Libraries Used
- Python
- Streamlit
- TensorFlow / Keras
- Pandas / NumPy
- yFinance API
- Matplotlib

---

## 📁 Project Structure
```
.
├── app.py
├── Stock Predictions Model.keras
├── Stock_Market_Prediction_Model_Creation.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 How to Run the Project
```bash
# 1. Clone the repository:
git clone https://github.com/your-username/stock-market-predictor.git

# 2. Navigate into the project folder:
cd stock-market-predictor

# 3. Install dependencies:
pip install -r requirements.txt

# 4. Run the app:
streamlit run app.py
```

---

## 📈 Model Details
The model uses 4 stacked LSTM layers with dropout regularization. 
Input shape is based on a 100-day window of past closing prices, and the output is the predicted next-day price.

---

## ✅ To Do / Future Enhancements
- Add reinforcement learning for dynamic adjustment
- Deploy with Docker or Streamlit Cloud
- Add user-uploaded CSV support

---

## 🙋‍♂️ Author
Developed by **Shrivardhan**  
Feel free to connect on LinkedIn or GitHub!
linkedin-Shrivardhan Bangale