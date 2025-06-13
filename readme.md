# 🌿 Skin Disease Detection and Recommendation System

This Streamlit application lets you **upload an image of a skin condition and instantly receive a detailed analysis and lifestyle recommendations** — all powered by a powerful Large Language Model (LLM) API.

---

## 🚀 Features

✅ Detects the skin disease from the photo  
✅ Shows its potential **causes**, **recommended drugs**, **natural treatments**, **specific diet and routine modifications**, and **skincare products with online purchase links**  
✅ Compresses and converts the photo to base64 efficiently (under 180 KB)  
✅ Integrates with **Nvidia's API** (Gemma-3-27B) for fast, accurate responses  

---

## 🛠 Tech Stack

- **Python 3.7+**
- **Streamlit** for UI
- **Pillow (PIL)** for image processing
- **Base64** for image compression and encoding
- **Nvidia API (Chat)** for large scale disease detection and recommendations
- **Requests** for API calls

---

## 📥 Installation

1️⃣ **Clone this repository:**

```bash
git clone https://github.com/your-username/skin-disease-detection.git
cd skin-disease-detection
````

2️⃣ **Install required packages:**

```bash
pip install streamlit Pillow requests
```

---

## 🔐 API Key

To use this, you'll need:

* An **Nvidia API key**

➥ Signup and retrieve your API key from [https://developer.nvidia.com/](https://developer.nvidia.com/)

➥ Update the following in `app.py`:

```python
api_key = "your_api_key_here"
```

---

## ⚙ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

Then view the application at:

```
http://localhost:8501
```

---

## 📝 How to Use

✅ **Step 1:** Upload a clear photo of the affected skin area.
✅ **Step 2:** The application converts and compresses the photo for API submission.
✅ **Step 3:** Once you click **Analyze**, it calls the API with your photo and prompt.
✅ **Step 4:** The results — including disease, causes, medication, lifestyle, and products — appear instantly on your screen.

---

---

## 🕹 Notes

* **Quality:** The photo should be clear, well-lighted, and show the affected area in detail.
* **Size:** Large images are compressed to under 180 KB for faster processing.
* **API:** The application uses the Gemini-3-27B endpoint by NVIDIA; make sure you have a valid API key.

---

## 🌟 Forward-Looking View

This pipeline can be easily integrated into:

✅ **Healthcare apps**
✅ **Dermatologists’ platforms**
✅ **Pharmacy services**
✅ **AI-assisted diagnostics**

---

## 📝 License

This project is licensed under the **MIT License** — see `LICENSE` for details.


