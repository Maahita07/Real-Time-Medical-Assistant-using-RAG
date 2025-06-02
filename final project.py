import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

# Page Config
st.set_page_config(page_title="AI Medical Assistant", layout="wide")

# Styling
st.markdown("""
    <style>
        body { font-family: 'Helvetica Neue', sans-serif; }
        .message-container { padding: 15px; border-radius: 15px; margin-bottom: 10px; font-size: 16px; max-width: 70%; }
        .user-message { background-color: #e1f7d5; color: #388e3c; align-self: flex-end; }
        .ai-message { background-color: #cfe2f3; color: #1e3a8a; align-self: flex-start; }
        .stButton button { background-color: #007bff; color: white; border-radius: 10px; }
        .header { color: #4caf50; font-weight: bold; font-size: 24px; }
        .button { border-radius: 12px; background-color: #007bff; color: white; padding: 12px 30px; font-size: 16px; transition: 0.3s ease; }
        .button:hover { background-color: #0056b3; }
    </style>
""", unsafe_allow_html=True)

# Adding hover effects to the buttons

# Title
st.title("🩺 Rapid Medical Insights (RMI)")
st.caption("Smarter, Context-Aware AI Medical Assistant")

# Session State
if "page" not in st.session_state:
    st.session_state.page = "Medical Assistant"
if "message_log" not in st.session_state:
    st.session_state.message_log = [
        {"role": "ai", "content": "👋 Hello! I'm your AI Medical Consultant. How can I assist you today?"}]

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.subheader("Model")
    rag_model = st.radio(
        "Select RAG Version",
        ["RAG 1 Model", "RAG 2 GB Model", "RAG 3 GB Model"],
        index=0
    )
    st.divider()
    st.subheader("Navigation")
    st.button("Medical Assistant", on_click=lambda: st.session_state.update(page="Medical Assistant"))
    st.button("Health Risk Calculator", on_click=lambda: st.session_state.update(page="Health Risk Calculator"))
    st.button("Cancer Risk Assessment", on_click=lambda: st.session_state.update(page="Cancer Risk Assessment"))
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai) + [LangChain](https://python.langchain.com)")
    st.markdown("Made by Maahita, Pushpek, Happy, and Zahids ✨")

# Model and VectorStore
selected_model = "deepseek-r1:1.5b"
embeddings = OllamaEmbeddings(base_url="http://127.0.0.1:11434", model=selected_model)
vectorstore = FAISS.from_texts(["Placeholder medical knowledge base."], embeddings)
llm_engine = ChatOllama(model=selected_model, base_url="http://127.0.0.1:11434", temperature=0.3)


# --- Functions ---
from langchain.chains import ConversationalRetrievalChain

# Build stronger system prompt
def dynamic_system_prompt(user_message):
    """Create smarter system prompts based on the user's recent message."""
    if any(term in user_message.lower() for term in ["cancer", "tumor", "oncology"]):
        return "You are an expert oncology assistant. Please respond empathetically, offer clear guidance, and suggest preventive actions and treatment options."
    elif any(term in user_message.lower() for term in ["heart", "stroke", "blood pressure"]):
        return "You are a cardiology assistant. Provide clear, evidence-based advice, and suggest lifestyle changes or further medical testing if needed."
    elif any(term in user_message.lower() for term in ["diabetes", "glucose", "insulin"]):
        return "You are an endocrinology assistant. Offer advice related to diabetes management, preventive tips, and lifestyle recommendations."
    else:
        return "You are a general healthcare AI. Provide clear, empathetic advice and suggest next steps based on the query."

# Create Conversational RAG Chain
def create_rag_chain():
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_engine,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )
    return qa_chain

# Main function to process user input
def process_user_query(user_query):
    rag_chain = create_rag_chain()
    system_prompt = dynamic_system_prompt(user_query)

    with st.spinner("🔍 Fetching best medical advice..."):
        result = rag_chain.invoke({
            "question": user_query,
            "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.message_log]
        })

    ai_response = result['answer']
    sources = "\n\n".join([doc.metadata.get('source', '') for doc in result.get('source_documents', [])])

    full_response = (
        f"## 📚 Context Retrieved\n{sources or 'No external references found.'}\n\n"
        f"---\n\n"
        f"## 🩺 RMI Assistant's Advice\n{ai_response}\n\n"
        f"---\n\n"
        f"💬 **Would you like more information about prevention or treatment options?**"
    )

    return full_response

def build_prompt_chain():
    """Build the conversation chain including memory context."""
    prompt_sequence = [
        SystemMessagePromptTemplate.from_template(dynamic_system_prompt(st.session_state.message_log[-1]['content']))]
    for msg in st.session_state.message_log[-5:]:  # only last 5 for brevity
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        else:
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)


def generate_ai_response(prompt_chain, context):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    response = processing_pipeline.invoke({"context": context})

    final_response = (
        f"## 📚 Context from Medical Knowledge Base\n{context}\n\n"
        f"---\n\n"
        f"## 🩺 RMI Response\n{response}\n\n"
        f"---\n\n"
        f"💬 **Would you like me to also suggest preventive measures or explain treatment options?**"
    )
    return final_response


# --- Pages ---
if st.session_state.page == "Medical Assistant":
    st.subheader("🤖 Ask a Medical Question")
    chat_container = st.container()

    with chat_container:
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        for message in st.session_state.message_log:
            msg_class = "user-message" if message["role"] == "user" else "ai-message"
            st.markdown(f'<div class="message-container {msg_class}">{message["content"]}</div>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    user_query = st.chat_input("Type your question...")

    if user_query:
        st.session_state.message_log.append({"role": "user", "content": user_query})
        with st.spinner("🔍 Analyzing your medical query with expert-level depth..."):
            # Retrieve relevant documents
            docs = vectorstore.similarity_search(user_query, k=5)
            context = "\n".join([doc.page_content for doc in docs]) if docs else "No external documents found."

            prompt_chain = build_prompt_chain()
            ai_response = generate_ai_response(prompt_chain, context)

        st.session_state.message_log.append({"role": "ai", "content": ai_response})
        st.rerun()

elif st.session_state.page == "Cancer Risk Assessment":
    st.subheader("🩺 Cancer Risk Checker")
    age = st.number_input("Your Age:", min_value=0, max_value=120, step=1)
    gender = st.radio("Gender:", ["Male", "Female"])
    family_history = st.selectbox("Family History of Cancer?", ["No", "Yes"])
    smoking = st.selectbox("Smoking?", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Consumption?", ["No", "Yes"])
    diet = st.selectbox("Healthy Diet?", ["Yes", "No"])
    exercise = st.selectbox("Regular Exercise?", ["Yes", "No"])
    exposure = st.selectbox("Exposure to Carcinogens?", ["No", "Yes"])
    medical_history = st.selectbox("Previous Medical History?", ["No", "Yes"])

    if st.button("🔍 Assess Risk"):
        risk_score = sum([
            age > 50,
            family_history == "Yes",
            smoking == "Yes",
            alcohol == "Yes",
            diet == "No",
            exercise == "No",
            exposure == "Yes",
            medical_history == "Yes"
        ]) * 2

        if risk_score >= 12:
            risk_level = "🔴 High Risk (70-90%)"
        elif risk_score >= 6:
            risk_level = "🟡 Moderate Risk (30-70%)"
        else:
            risk_level = "🟢 Low Risk (10-30%)"

        st.success(f"Estimated Cancer Risk: **{risk_level}**")
        st.info("For a detailed assessment, consult your healthcare provider.")

elif st.session_state.page == "Health Risk Calculator":
    st.subheader("🧮 Health Risk Estimator")
    gender = st.radio("Gender:", ["Male", "Female"])
    age = st.number_input("Age:", min_value=0, max_value=120, step=1)
    weight = st.number_input("Weight (kg):", min_value=0.0, step=0.1)
    height = st.number_input("Height (cm):", min_value=0.0, step=0.1)
    smoking = st.selectbox("Smoking Frequency:", ["No", "Occasionally", "Regularly"])
    exercise = st.selectbox("Exercise Habit:", ["Rarely", "1-2 times/week", "3-5 times/week", "Daily"])
    bp = st.selectbox("High Blood Pressure?", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes?", ["No", "Yes"])

    if st.button("🔎 Calculate Health Risk"):
        bmi = weight / (height / 100) ** 2
        risk_points = sum([
            bmi > 30,
            smoking == "Regularly",
            exercise == "Rarely",
            age > 50,
            bp == "Yes",
            diabetes == "Yes"
        ]) * 2

        if risk_points >= 8:
            risk_category = "🔴 High Risk"
            suggestion = "Focus on lifestyle changes immediately: healthy diet, exercise, smoking cessation."
        elif risk_points >= 4:
            risk_category = "🟡 Moderate Risk"
            suggestion = "Improve habits: Regular exercise, routine medical check-ups."
        else:
            risk_category = "🟢 Low Risk"
            suggestion = "Maintain current healthy lifestyle!"

        st.success(f"Your Health Risk Level: **{risk_category}**")
        st.info(f"Recommendation: {suggestion}")

