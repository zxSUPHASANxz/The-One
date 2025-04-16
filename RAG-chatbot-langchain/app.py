from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, pipeline
import gradio as gr
import os
from dotenv import load_dotenv, find_dotenv

# โหลด token ถ้ามี แต่ไม่จำเป็นสำหรับเวอร์ชันนี้
_ = load_dotenv(find_dotenv())

# ✅ STEP 1: โหลดข้อมูลจาก URL
url = "https://www.who.int/news-room/fact-sheets/detail/coronavirus-disease-(covid-19)"
loader = WebBaseLoader(url)
data = loader.load()

# ✅ STEP 2: แยกข้อความเป็น chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(data)

# ✅ STEP 3: ใช้ Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# ✅ STEP 4: สร้างเวกเตอร์ index
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})

# ✅ STEP 5: ตั้งค่า QA pipeline
tokenizer = AutoTokenizer.from_pretrained("Intel/dynamic_tinybert", padding=True, truncation=True, max_length=512)

question_answerer = pipeline(
    "question-answering",
    model="Intel/dynamic_tinybert",
    tokenizer=tokenizer,
    return_tensors='pt'
)

# ✅ STEP 6: ฟังก์ชันตอบคำถาม
def generate(question):
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])
    squad_ex = question_answerer(question=question, context=context)
    return squad_ex['answer']

def respond(message, chat_history):
    bot_message = generate(message)
    chat_history.append((message, bot_message))
    return "", chat_history

# ✅ STEP 7: สร้าง UI ด้วย Gradio
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240)
    msg = gr.Textbox(label="Ask away")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

demo.queue().launch()