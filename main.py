from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
from pinecone import Pinecone as pc
from langchain.chains import RetrievalQA
from dotenv import find_dotenv, load_dotenv
import streamlit as st
import os
import warnings 
warnings.filterwarnings('ignore')

load_dotenv()


# Configure Streamlit page settings
st.set_page_config(
    page_title="Chat with Gemini-Pro!",
    page_icon="🤖",  # Favicon emoji
    layout="centered",  # Page layout option
)

GOOGLE_API_KEY='AIzaSyB-C5_2h5nQfIBUYKjBxKs_m55lWTRDnRg'
PINECONE_API_KEY='5471e559-8220-4014-ac6a-9620a5172f3a'


st.title("Courses RAG 📚")

pc = pc(api_key=PINECONE_API_KEY)
index = pc.Index("mychatpot")

def get_answer(query):

    model = 'models/embedding-001'
    embed = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY,model=model)

    vstore = Pinecone.from_existing_index(embedding=embed,index_name='mychatpot')
    retriever = vstore.as_retriever(k=10)

    llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.7 ,convert_system_message_to_human=True)

    template = """You play the role of an assistant who will help students to find courses suited to them. A student is going to ask you a question, your task is to answer the question using the context. if the question is not related to the context, try to guide the student to ask you about Coursera courses. \n\n
                    Context: \n {context}?\n
                    Question: \n {question} \n
                    Answer:
                  """

    QA_CHIAN_PROMPT = PromptTemplate.from_template(template)

    qa_chian = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={
            'prompt':QA_CHIAN_PROMPT
        }
    )
    response = qa_chian.invoke(query)
    return response['result']

input = st.text_input('input: ',key = 'input')
submit = st.button(" 🆗 ")

if submit:
    response = get_answer(input)
    st.write(response)

footer="""
<style>
        .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        }
</style>
<div class="footer">
<p>For Feedback: 
<a href="https://www.linkedin.com/in/zaid-allwansah/" target="_blank">Zaid</a> |
<a href="https://www.linkedin.com/in/mohammad-aljermy-139b6b24a/" target="_blank">Mohammad Aljermy</a> |
<a href="https://www.linkedin.com/in/mahmood-abusaa-311389263?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank">Mahmoud Abu saa</a>

</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
