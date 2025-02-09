# **Multi-User-Document-Search-and-Conversational-Q-A-System**
This system is a web-based multi-user document search application that allows users to securely search through their restricted-access documents and interact with an AI-powered conversational Q&A system. The system utilizes hybrid Search (semantic search + Keyword Search) powered by sentence transformers and integrates query summarization to provide more insightful responses.

## **Features**

* Secure Multi-User Authentication: Users can access only the documents assigned to them.

* Document Upload & Storage: Pre-loaded documents stored in a dedicated folder.

* Hybrid Search Approach: Combines keyword matching and semantic search for better accuracy.

* AI-Powered Conversational Q&A: Maintains context-aware responses using previous conversation history.

* Document Summarization: Extracts relevant excerpts and summarizes them for a concise answer.

* Follow-up Query Support: Users can refine their searches and receive context-aware responses.

## **Technologies Used**

+ Frontend: Streamlit

+ Backend: Python

###  NLP Models:

- all-MiniLM-L6-v2 (Sentence Transformer for semantic search)

- facebook/bart-large-cnn (Summarization model)


### Libraries:

* streamlit (UI framework)

* sentence-transformers (Semantic similarity search)

* torch (Deep learning framework)

* transformers (Hugging Face models)

* PyPDF2 (PDF text extraction)

* pandas (Data handling)

## **Folder Structure**

📂 project_root/
 ├── 📂 uploaded_docs/      # Folder containing uploaded PDF documents
 ├── app.py                # Main Streamlit application
 ├── requirements.txt      # List of dependencies
 ├── README.md             # Documentation
 

## **Setup Instructions**

1️. Install Dependencies
pip install -r requirements.txt

2. Upload Documents
Place the PDF documents inside the uploaded_docs/ folder before running the app.

3. Run the Application
streamlit run app.py

## **User Authentication**

Users must enter their email to access their assigned documents. The USER_ACCESS dictionary in the code manages document access:

## **How It Works**

1️. User Login: Users enter their email to authenticate.
2️. Document Access: The system loads the documents assigned to the user.
3️. Search & Retrieval: Uses semantic search and keyword matching to retrieve the most relevant excerpts.
4️. Summarization & Contextual Responses: Uses an AI-powered summarization model to generate concise answers.Maintains conversation history for improved follow-up queries.

## **Future Enhancements**

✅ User Upload Feature (Allow users to upload their own PDFs dynamically)

✅ Advanced Query Expansion (Use LLMs for enhanced understanding of user queries)

✅ Database Integration (Store user authentication and documents in a database)
 
