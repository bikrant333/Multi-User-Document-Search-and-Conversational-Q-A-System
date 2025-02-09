import os
import re
import streamlit as st
import pandas as pd
import json
from collections import defaultdict
from typing import List, Dict
import openai
from sentence_transformers import SentenceTransformer, util
import torch
import PyPDF2
from transformers import pipeline
from keybert import KeyBERT

# Load the model for semantic search
model = SentenceTransformer("all-MiniLM-L6-v2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Simulated user database with restricted document access
USER_ACCESS = {
    "alice@email.com": ["accenture-fourth-quarter-fiscal-2024.pdf"],
    "bob@email.com": ["happiest_minds_EarningsCall.pdf", "HDFC_bank_earning_calls.pdf"],
    "charlie@email.com" : ["TCS_Q3 2024-25 Earnings Conference Call.pdf", "wipro-earnings.pdf"],
}

DATA_FOLDER = "uploaded_docs"  # Folder where documents are stored

# Function to load documents from local storage
def load_documents() -> Dict[str, str]:
    documents = {}
    for user_docs in USER_ACCESS.values():
        for doc in user_docs:
            doc_path = os.path.join(DATA_FOLDER, doc)
            if os.path.exists(doc_path):
                try:
                    with open(doc_path, "rb") as file:
                        reader = PyPDF2.PdfReader(file)
                        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
                        documents[doc] = text
                except Exception as e:
                    documents[doc] = f"Error reading {doc}: {str(e)}"
    return documents

# Load all documents at runtime
DOCUMENTS = load_documents()

if "query" not in st.session_state:
    st.session_state.query = ""

# Store conversation history for each user
if "session_history" not in st.session_state:
    st.session_state.session_history = defaultdict(list) # Store chat history per user

def authenticate_user(email: str) -> bool:
    return email in USER_ACCESS

def retrieve_documents(email: str) -> Dict[str, str]:
    accessible_docs = USER_ACCESS.get(email, [])
    return {doc: DOCUMENTS[doc] for doc in accessible_docs if doc in DOCUMENTS}


def hybrid_search(query: str, documents: Dict[str, str], context: List[str]) -> str:
    """Find the most relevant document excerpts for the given query using semantic search, incorporating past interactions."""
    kw_model = KeyBERT()

    if not documents:
        return "Unauthorized document"
    doc_texts = list(documents.values())
    conversation_context = " ".join(context[-3:])  # Include last 5 interactions for context
    query_with_context = conversation_context + " " + query

    # Tokenize sentences
    doc_sentences = [sentence.strip() for doc in doc_texts for sentence in re.split(r'(?<=[.!?])\s+', doc) if sentence.strip()]
    if not doc_sentences:
        return "No relevant content found."
    
    # Keyword Search: Find sentences containing the query term(s)
    query_lower = query.lower()
    query_keywords = kw_model.extract_keywords(query_lower, keyphrase_ngram_range=(1,2), stop_words='english')

    keyword_matches = [sentence for sentence in doc_sentences if query_keywords[0][0] in sentence.lower()]
    
    # Semantic Search: Compute similarity scores
    query_embedding = model.encode(query_with_context, convert_to_tensor=True)
    doc_embeddings = model.encode(doc_sentences, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]

    # Get top 3 matches instead of just 1
    top_indices = torch.topk(scores, 3).indices.tolist()

    # Combine keyword and semantic search results
    result_sentences = set()
    if not keyword_matches:
        return "No relevant content found."
    else:
        result_sentences.update(keyword_matches[:2])  # Top keyword matches

    for idx in top_indices:
        excerpt = " ".join(doc_sentences[max(0, idx-2):idx+2])  # Get a small window around top matches
        if excerpt not in result_sentences:
            result_sentences.add(excerpt) # Top semantic matches

    return " ".join(result_sentences) if result_sentences else "No relevant content found."
    # return " ".join(doc_sentences[max(0, best_match_idx-5):best_match_idx+5])  # Extract a larger context window

def retrieve_relevant_excerpts(query: str, documents: Dict[str, str]) -> List[str]:
    """Retrieve multiple relevant excerpts from documents based on the query."""
    if not documents:
        return "Unauthorized document"
    doc_texts = list(documents.values())
    doc_sentences = [sentence.strip() for doc in doc_texts for sentence in re.split(r'(?<=[.!?])\s+', doc) if sentence.strip()]
    if not doc_sentences:
        return ["No relevant excerpts found."]
    
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode(doc_sentences, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]

    top_indices = torch.topk(scores, 2).indices.tolist()
    
    excerpts = []
    for idx in top_indices:
        excerpt = " ".join(doc_sentences[max(0, idx-5):idx+5])  # Get larger context window
        excerpts.append(excerpt)
        if excerpt not in excerpts:  # Avoid repetition
            excerpts.append(excerpt)
    
    return excerpts

def generate_summary(excerpt: str, context: List[str] = []) -> str:

    """Generate a summary of the extracted excerpt incorporating previous context."""

    context_text = " ".join(context[-3:])  # Include last 3 exchanges for summary
    full_text = context_text + " " + excerpt    #Include context
    summary = summarizer(full_text, max_length=95, min_length=25, do_sample=False)
    return summary[0]['summary_text']


# Streamlit UI
st.title("Multi-User Document Search System")

email = st.text_input("Enter your email to log in:")
if email:
    if authenticate_user(email):
        st.success(f"Logged in as {email}")
        user_docs = retrieve_documents(email)
        st.write("You have access to the following documents:")
        for doc in user_docs.keys():
            st.write(f"- {doc}")

        if email not in st.session_state.session_history:
            st.session_state.session_history[email] = []

        # query = st.text_input("Enter your query:", key=f"query_{email}")

        with st.form(key="query_form",clear_on_submit=True):
            query = st.text_input("Enter your query:", key=f"query_{email}")
            submit = st.form_submit_button(label='Submit')

        if submit:

            conversation_context = [entry[1] for entry in st.session_state.session_history[email][-5:]]
            relevant_excerpt = hybrid_search(query, user_docs, conversation_context)
            if relevant_excerpt =="No relevant content found.":
                st.write("### No relevant content found in user accessed documents ###")
            else:
                summary = generate_summary(relevant_excerpt, conversation_context)
                
                relevant_excerpts = retrieve_relevant_excerpts(query, user_docs)
                st.session_state.session_history[email].append(("User", query))
                st.session_state.session_history[email].append(("BOT", summary))
                
                st.write("### Relevant Excerpts:")
                # for excerpt in relevant_excerpt:
                #     st.write(f"- {excerpt}")
                st.write(f"- {relevant_excerpt}")
                st.write("### Conversation:")
                for speaker, text in st.session_state.session_history[email]:
                    st.write(f"**{speaker}:** {text}")    # Clear the input after submission



        # follow_up_query = st.text_input("Ask a follow-up question:", key=f"followup_{email}")

        with st.form(key="follow_up_query_form",clear_on_submit=True):
            follow_up_query = st.text_input("Ask a follow-up question:", key=f"followup_{email}")
            submit = st.form_submit_button(label='Submit')

        if follow_up_query:
            follow_up_context = [entry[1] for entry in st.session_state.session_history[email][-5:]]
            relevant_excerpt1 = hybrid_search(follow_up_query, user_docs, follow_up_context)
            follow_up_summary = generate_summary(relevant_excerpt1, follow_up_context)
            
            relevant_excerpts = retrieve_relevant_excerpts(follow_up_query, user_docs)
            
            st.session_state.session_history[email].append(("User", follow_up_query))
            st.session_state.session_history[email].append(("BOT", follow_up_summary))
            
            st.write("### Relevant Excerpts:")
            for excerpt in relevant_excerpts:
                st.write(f"- {excerpt}")
            
            st.write("### Conversation::")
            for speaker, text in st.session_state.session_history[email]:
                st.write(f"**{speaker}:** {text}")

    else:
        st.error("Unauthorized email. Access denied.")
