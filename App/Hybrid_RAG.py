from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain_groq import ChatGroq
import os

import dotenv
dotenv.load_dotenv()

# --- Data Ingestion ---

def ingest_pdf(file_path):
    elements = partition_pdf(
        file_path,
        strategy="hi_res",
        extract_images_in_pdf=True
    )
    return elements

# --- Content Processing ---

def table_to_text(table):
    rows = getattr(table, "rows", None)
    if not rows:
        return ""
    lines = ["\t".join(str(cell) for cell in row) for row in rows]
    return "\n".join(lines)

def process_content(elements):
    combined_text = []
    images = []
    for element in elements:
        ele_type = type(element).__name__
        if ele_type not in ["Table", "Image", "Formula"]:
            if hasattr(element, "text") and element.text:
                combined_text.append(element.text)
        elif ele_type == "Formula":
            if hasattr(element, "text") and element.text:
                combined_text.append(element.text)
        elif ele_type == "Image":
            images.append(element)
        elif ele_type == "Table":
            combined_text.append(table_to_text(element))
    return combined_text, images

# --- Content Chunking ---

def chunk_text(content, chunk_size=500, chunk_overlap=50):
    combined_text = "\n".join(str(c) for c in content)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_text(combined_text)
    return chunks

# --- Embedding & Storage ---

def embed_and_store(chunks, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    vector_store = Chroma.from_texts(
        chunks, 
        embedding_model,
    )
    return vector_store

# --- Hybrid Retriever ---

def hybrid_retriever(content, vector_store):
    docs = [Document(page_content=c) for c in content]
    keyword_retriever = BM25Retriever.from_documents(docs)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[keyword_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

# --- RAG Chain Creation ---

def conversational_rag(retriever):
    
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.5)

    contextualize_q_system_prompt = """
        Given the chat history and the latest user question,
        formulate a standalone question that can be understood without the chat history.
        Do NOT answer the question, just reformulate it if needed.
        Consider that the documents may contain:
        - Complex tables with structured data
        - Mathematical formulas and equations
        - Code snippets and technical diagrams
        - Images with important visual information

        Your reformulated question should:
        - Include relevant details and references from the chat to be fully clear on its own.
        - Preserve the user's original intent and specificity.
        - Resolve pronouns or ambiguous terms using the chat history.
        - Avoid adding new information or answering the question.
        - Be concise and to the point.
    """

    contextualize_q_prompt = ChatPromptTemplate([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """
        You are an expert assistant specialized in analyzing complex documents.

        The context may include:
        - Tables with structured and relational data (analyze patterns, relationships, and key metrics)
        - Mathematical formulas and equations (explain concepts clearly and accurately)
        - Code snippets and programming logic (describe functionality and purpose)
        - Technical diagrams and visual information (interpret descriptions and relevant details)

        Instructions:
        - Your task is to answer the user's question based on the provided context. Refer chat history if needed.
        - If the provided context or chat history doesn't contain the answer to the user's question, mention that 
          "the context doesn't contain anything relevent to the question" and ask if the user wants you to give an answer from the external sources.
          and if the user agrees then you will be giving the answer based on your own knowledge and formulate a best fitting answer to that question.
        - In case if the context or the chat history doesn't contain the answer and you asked if the user wants you to give an answer from the external sources
          and the user agreed and you have to formulate an answer, do not hallucinate. Only give the answer if you 
          actually know it. Otherwise just say that the context doesn't contain anything relevent to the question and
          you couldn't find the answer from extrenal sources.
        - If the user explicitly asks you to give an answer from the extrenal sources then do it. But do not hellucinate. 
          Only give the answer if you actually know it. Otherwise tell the user that you couldn't find the answer from extrenal sources.
        - Provide clear, well-organized, and comprehensive answers.
        - Reference explicitly which part of the context supports your answer, such as "According to the provided table..."
          or "As shown in the formula..."
        - Avoid speculation or adding unsupported information.
        - Use technical terminology appropriately while keeping explanations accessible.

        Context:
        {context}

        Chat History:
        {chat_history}

        Question:
        {input}

        Answer:
    """

    qa_prompt = ChatPromptTemplate([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

# --- Chat History Formatting ---

def format_chat_history(chat_history):
    messages = []
    for question, answer in chat_history:
        messages.extend([
            HumanMessage(content=question),
            AIMessage(content=answer)
        ])
    return messages
