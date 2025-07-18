# Medical Chatbot with RAG and Qdrant

This project is a medical chatbot using Retrieval-Augmented Generation (RAG) with Qdrant vector database and BAAI/bge_large_en embeddings. It stores medical Q&A data for efficient search and response generation.

---

## Features

- Uses BAAI/bge-large_en embedding model .
- Stores embeddings and data in Qdrant Cloud vector database .
- Retrieval augmented generation with LangChain and Ollama LLM .
- Has an appointment booking feature .
- Uses Radicale for appointment booking .
- Fully open_source .
- Easy to deploy and extend

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yesmineChaari/Medical-Chatbot.git
cd Medical-Chatbot
```
### 2. Install  dependencies
```bash
pip install -r requirements.txt

```
### 3.Configure environment variables
Create a .env file in the root directory to store your Qdrant API key and URL securely.

Example .env file:
   QDRANT_API_KEY=your_qdrant_api_key_here
   QDRANT_URL=https://your-qdrant-cloud-url
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USERNAME=user.email@gmail.com
   SMTP_PASSWORD= *******
   FROM_EMAIL=user.email@gmail.com
   RADICALE_URL=http://localhost:5232/
   USERNAME=username
   PASSWORD=password


### 4. Running the application
First, install docker and set it up.
 Run the qdrant_bge_m3.py script to create and populate the vector database that will be used for RAG:
 ```bash
python build_qa_vectors_BAAI_bge_large_en.py
```

### 5. Installing and configuring Radicale
 ```bash
python -m pip install --upgrade https://github.com/Kozea/Radicale/archive/master.tar.gz
python -m radicale --storage-filesystem-folder=~/radicale/collections --auth-type none
install thunderbird 
```

### 6.Running the application
Run the streamlit_app.py 
 ```bash
streamlit run streamlit_app.py
```



