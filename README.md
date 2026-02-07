# 🌍 GeoDetector - Pinpoint The Unknown

**GeoDetector** is an intelligent AI-powered platform designed to revolutionize Cartographic Quality Assurance. It combines advanced Computer Vision with Generative AI to automatically detect map distortions and act as an expert cartography assistant.

![GeoDetector Banner](static/images/logo.png)

## 🚀 Key Features

### 1. 🔍 Instant Map Analysis (The Eye)
*   **AI Engine:** Uses a custom-trained **ResNet-18** Deep Learning model (PyTorch).
*   **Function:** instantly scans map images to detect geometric distortions, labeling errors, and poor topology.
*   **Smart Filtering:** Includes heuristic checks to reject non-map images (e.g., dark photos, random objects) before AI processing.
*   **Voice Feedback:** The system **speaks out the result** ("Good Map" or "Bad Map") using Test-to-Speech API.

### 2. 🤖 GeoBot Assistant (The Brain)
*   **Powered by:** **Llama 3.2** (via Ollama).
*   **Role:** An expert Cartography Consultant.
*   **Capabilities:**
    *   Explains *why* a map was flagged as "Bad".
    *   Provides technical advice on fixing river labeling, projection errors, and more.
    *   **Voice Interaction:** GeoBot greets you and **speaks its answers** out loud.

### 3. 🎨 Premium User Interface
*   **Modern Design:** Built with **TailwindCSS** and Glassmorphism effects.
*   **Interactive 3D Background:** Features a rotating wireframe terrain using **Three.js**.
*   **Smooth Animations:** Professional loading screens and transition effects.

---

## 🛠️ Tech Stack

*   **Backend:** Python, Django 4.2
*   **AI/ML:** PyTorch, Torchvision, Scikit-learn
*   **LLM:** Ollama (Llama 3.2)
*   **Frontend:** HTML5, TailwindCSS, Vue.js (Lightweight), Three.js
*   **Deployment:** Ready for Render / Heroku with `gunicorn` & `whitenoise`.

---

## ⚙️ Installation & Setup

### Prerequisites
*   Python 3.10+
*   [Ollama](https://ollama.com/) installed and running (`ollama serve`).
*   Model weights `best_map_model2.pth` placed in the root directory.

### 1. Clone the Repository
```bash
git clone https://github.com/Manish9211Ram/GeoDetector.git
cd GeoDetector
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Ollama (for Chatbot)
Ensure you have the Llama 3.2 model pulled:
```bash
ollama pull llama3.2
```

### 4. Run the Application
You can use the convenient batch script on Windows:
```bash
start.bat
```
Or run manually:
```bash
python manage.py runserver
```

Visit the app at: `http://127.0.0.1:8000`

---

## 📸 Screenshots

| Dashboard | Analysis Result |
|-----------|-----------------|
| *(Add screenshot here)* | *(Add screenshot here)* |

---

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

---

## 📜 License
This project is licensed under the MIT License.
