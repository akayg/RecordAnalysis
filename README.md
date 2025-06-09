# 📄 RecordAnalysis

**RecordAnalysis** is a Streamlit-based AI assistant designed to act as a **virtual doctor**, answering user questions based on a given **medical context**. It leverages **Google's Gemini (Generative AI)** via LangChain to provide detailed, relevant medical answers, and clearly handles irrelevant or ambiguous queries with predefined responses.

---

## 🧠 Features

- 🩺 **AI Doctor Assistant**: Interprets and responds to health-related queries using Google’s Generative AI.
- 📄 **PDF Medical Document Support**: Users can upload documents (e.g., patient reports, research papers) to serve as context.
- 💬 **Natural Language Understanding**: Built on LangChain + Gemini API for coherent, context-aware answers.
- ✅ **Context-Aware Behavior**:
  - If the query is **unrelated to medicine**, the model responds:  
    `Context is not related to medical.`
  - If the **answer isn't found** in the given context, it responds:  
    `I CAN'T GET IT! Please Rephrase the question.`

---

## 🧰 Tech Stack

| Component               | Description                                       |
|------------------------|---------------------------------------------------|
| 🐍 Python              | Primary language                                  |
| 📦 Streamlit           | Frontend web UI framework                         |
| 🤖 Google Generative AI | For intelligent and contextual responses          |
| 🧠 LangChain           | Manages prompt templates and context chaining     |
| 📚 PyPDF2              | Parses uploaded PDFs for analysis                 |
| 🧠 ChromaDB + FAISS    | Vector store for fast context retrieval           |
| 🔐 python-dotenv       | Loads API keys and env variables securely         |
| 📋 streamlit-option-menu | Sidebar navigation and UI interactions          |

---

## 📸 Screenshots

*(Add screenshots here if available)*

---

## 🚀 Getting Started

### 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/akayg/RecordAnalysis.git
cd RecordAnalysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 🗝️ Environment Setup

Create a `.env` file in the root directory and add your Google API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 🏃‍♂️ Run the App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
RecordAnalysis/
├── app.py                  # Streamlit UI and logic
├── utils/                  # Utility functions
│   ├── pdf_reader.py       # PDF parsing and cleaning
│   └── prompt_templates.py # Custom prompts for AI
├── .env                    # Your API keys (not pushed to GitHub)
├── requirements.txt        # All dependencies
└── README.md               # Project documentation
```

---

## ✅ Example Usage

1. Upload a medical document (PDF).
2. Ask a question related to the document.
3. Get a contextual and medically detailed answer.

> 🧠 Example Prompt:  
> “What are the common symptoms of Type 2 Diabetes mentioned in the uploaded report?”

> ✅ Response (if found in context):  
> “According to the document, common symptoms include increased thirst, frequent urination, fatigue…”

> ❌ If the question is off-topic:  
> `Context is not related to medical.`

> ❓ If question is unclear or no match:  
> `I CAN'T GET IT! Please Rephrase the question.`

---

## 🤝 Contributing

Contributions are welcome!

1. Fork this repo
2. Create a new branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -am 'Added new feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 📬 Contact

- GitHub: [@akayg](https://github.com/akayg)
- Email: *(add your email if you'd like others to reach you)*
