# 🧠 MindNest – Mental Health Chatbot

MindNest is an **AI-powered mental health chatbot** that provides safe, private, and empathetic conversations for users seeking mental wellness support.  
Built with **Flask, PyTorch, Hugging Face Transformers, and Tailwind CSS**, this project integrates an ML model with a responsive frontend to deliver real-time chat experiences.  

---

## 📂 Project Structure

Mental Health Chatbot/
│── backend/
│ │── app.py # Flask app (serves frontend + API)
│ │── train_model.py # Model training script
│ │── chat_with_model.py # Local testing for model responses
│ │── mental_health_chatbot/ # Saved model + tokenizer
│ │── templates/ # Frontend pages
│ │ ├── index.html
│ │ ├── chat.html
│ │ └── about.html
│ │── static/ # Static assets (images, css, js)
│ ├── logo.png
│ ├── feature1.png
│ ├── feature2.png
│ ├── feature3.png
│ ├── testimonials.png
│ ├── linkedin.png
│ ├── github.png
│ ├── mail.png
│ └── ...
│
│── data/
│ └── NLP mental health.csv # Training dataset
│
│── README.md # Documentation
│── requirements.txt # Dependencies


---

## 🚀 Features
- 🗨️ **Real-time Chat** – Users can chat with the AI chatbot through an intuitive web interface.
- 🔐 **Privacy-Focused** – Conversations are not logged permanently; user safety first.
- 📚 **Evidence-Based** – Responses are generated using NLP trained on mental health conversations.
- 🎨 **Responsive UI** – Built with TailwindCSS for a clean, modern look.
- ⚡ **Flask API** – Backend integrates ML model with REST API endpoints.

---

## 🛠️ Tech Stack

- **Frontend:** HTML, TailwindCSS, JavaScript  
- **Backend:** Flask, Flask-CORS  
- **ML/NLP:** Hugging Face Transformers, PyTorch  
- **Deployment Options:** Render, Railway, Hugging Face Spaces, AWS/GCP  

---
### 1. Clone the repository
git clone https://github.com/your-username/mindnest-chatbot.git
cd mindnest-chatbot/backend

### 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run the Flask server
python app.py

---
### Dataset

Dataset: NLP Mental Health Conversations (data/NLP mental health.csv) 

Used for training a chatbot that can provide supportive and empathetic responses.

## 👨‍💻 Developer

**Suyash Kumar Gupta**  
AI & Data Science Enthusiast | Prompt Engineer | UX Researcher  

🔗 [LinkedIn](https://www.linkedin.com/in/suyashgupta0098/)  
🔗 [GitHub](https://github.com/Suyash-kr)  
📧 suyash0098@gmail.com  

## 📜 License
This project is licensed under the **MIT License**.  
Feel free to use, modify, and share it for educational purposes.  

---

⭐ If you like this project, don’t forget to **star the repo**!
