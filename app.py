import streamlit as st
import time
import hashlib
from utils import check_cache,save_to_cache,cleanup_cache_by_id
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

@st.cache_resource
def initialize_system():
    embed_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./vector_db", embedding_function=embed_model)

    cache_db=Chroma(
        persist_directory="./my_semantic_cache", 
        embedding_function=embed_model, 
        collection_name="qa_cache"
    )

    llm = ChatOllama(model="llama3.2:1b")

    return embed_model,vector_db,cache_db,llm

embed_model, vector_db, cache_db, llm = initialize_system()

template="""You are a professional and friendly career assistant. 
    Your goal is to answer questions about a candidate's skills based on their resume/profile data.
    your goal is to answer the question as if the candidate answers to the requiterer.

    RULES:
    1. Do NOT say "Based on the context" or "According to the document."
    2. Speak as if you already know the candidate well.
    3. Use a friendly, conversational, and professional tone.
    4. If the information isn't there, politely say you aren't sure about that specific detail.
    5. Keep it concise but enthusiastic.
    6. If you don't have enough information to answer, ask for clarification.
    7. if the question is ask about a job/role she fits in, i want you to answer accordingly , and also put the skills and relevant projects worked on.
    8. start the sentence like  "Lavanya did this, Lavanya did that, Lavanya is a professional in this field, Lavanya has worked on this project, Lavanya is skilled in this area."
    9. if the question is ask about a job/role she fits in, i want you to answer accordingly , and also put the skills and relevant projects worked on.

    When asked about PROJECTS:
1. Clearly state the Project Name.
2. Detail the "Domain" and "Technologies & Tools" used (e.g., Python, OpenCV, NLP).
3. Explain the specific "Skills Demonstrated".
4. Explicitly link the project to one of her target "Professional Roles" (e.g., AI Engineer, Data Scientist).

When asked about RESPONSIBILITIES:
1. List her core capabilities (ML, NLP, Computer Vision).
2. For each responsibility, give a concrete example from one of her projects as proof.(e.g Automatic attendance system , segmentation, car parking), and give a small info about that project

If the information is not in the context, say: "The provided resume does not contain details regarding that specific request."



    Candidate Information:
    {context}
    
    User Question: {question}
    Friendly Response:
   
    """

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": vector_db.as_retriever(search_kwargs={"k": 3}), "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

st.title("Question bot")
st.info("Ask a question below. I will check the semantic cache first, then search your PDF.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the candidate..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            # Use the imported functions!
            ans, score = check_cache(prompt, embed_model, cache_db)
            
            if ans:
                response = f"⚡ [Cache Hit - {score:.2f}]\n\n{ans}"
            else:
                response = rag_chain.invoke(prompt)
                save_to_cache(prompt, response, cache_db)
                cleanup_cache_by_id(cache_db) # Keep it tidy!
            
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})