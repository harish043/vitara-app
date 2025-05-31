import streamlit as st
import os
import json
import requests
import datetime
import traceback
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("🚨 GOOGLE_API_KEY is not set in the environment!")

# Constants
USAGE_FILE = "ip_usage.json"
QUERY_LIMIT = 50

# Load usage data
def load_ip_usage():
    if os.path.exists(USAGE_FILE):
        with open(USAGE_FILE, "r") as f:
            return json.load(f)
    return {}

# Save usage data
def save_ip_usage(data):
    with open(USAGE_FILE, "w") as f:
        json.dump(data, f)

# Get user IP
def get_ip():
    try:
        return requests.get('https://api.ipify.org').text
    except:
        return "unknown"

# Query tracking
ip_usage = load_ip_usage()
def is_allowed(ip):
    today = datetime.date.today().isoformat()
    if ip not in ip_usage:
        ip_usage[ip] = {today: 1}
    elif today not in ip_usage[ip]:
        ip_usage[ip][today] = 1
    elif ip_usage[ip][today] < QUERY_LIMIT:
        ip_usage[ip][today] += 1
    else:
        return False
    save_ip_usage(ip_usage)
    return True

# UI setup
st.set_page_config(page_title="Vitara", layout="centered")
st.title("🤖 Vitara – Your VIT College Assistant!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load vectorstore
@st.cache_resource
def get_vectorstore():
    db_dir = "./db"
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    return Chroma(persist_directory=db_dir, embedding_function=embeddings)

# Load LLM
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY
    )

# Prompt for Vitara
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are Vitara, an experienced college senior and assistant helping VIT Vellore students with college-related questions. "
        "Answer clearly, concisely, and empathetically. Be friendly and helpful, while maintaining professionalism. "
        "Only use the context if necessary, and do not mention it explicitly.\n\n{context}"
    )),
    ("human", "{input}")
])

# Build RAG chain
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = get_llm()

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# Chat input
query = st.chat_input("Ask me anything about VIT Vellore...")
if query:
    ip = get_ip()
    if not is_allowed(ip):
        st.error("🚫 Daily query limit reached for your IP. Try again tomorrow.")
    else:
        st.session_state.chat_history.append(("user", query))
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke({"input": query})
                answer = response["answer"]
            except Exception as e:
                answer = "⚠️ Sorry, I ran into an issue. Please try again."
                print(traceback.format_exc())
        st.session_state.chat_history.append(("bot", answer))

# Chat rendering
st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
for sender, msg in st.session_state.chat_history:
    class_name = "user-msg" if sender == "user" else "bot-msg"
    st.markdown(f"<div class='{class_name}'>{msg}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Light theme CSS
st.markdown("""
    <style>
    .user-msg {
        background-color: #dcf8c6;
        color: black;
        padding: 10px 15px;
        border-radius: 16px;
        max-width: 75%;
        margin-left: auto;
        margin-bottom: 10px;
    }
    .bot-msg {
        background-color: #f1f0f0;
        color: black;
        padding: 10px 15px;
        border-radius: 16px;
        max-width: 75%;
        margin-right: auto;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)
