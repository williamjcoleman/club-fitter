import streamlit as st
import langchain
from langchain_helper import get_qa_chain, create_vector_db

st.title("Club Fitter")
st.sidebar.header('Helpful Resources')
st.sidebar.subheader('Ping - https://ping.com/en-us/blogs/proving-grounds')
st.sidebar.subheader('Titleist - https://www.titleist.com/fitting')
st.sidebar.subheader('Mizuno - https://mizunogolf.com/us/custom-fit/')
st.sidebar.subheader('Callaway - https://www.callawaygolf.com/custom-fitting/')
st.sidebar.subheader('Taylormade - https://www.taylormadegolf.com/locations.html?lang=en_US')
st.sidebar.subheader('Trackman - https://blog.trackmangolf.com')
                     
st.markdown('''
        Welcome to your own personal AI Club Fitter. Ask any questions that come to mind like:  
            - What are the different flexes of shafts?  
            - What is an optimal launch angle for my driver?  
            - How do I know what lie angle I need on my irons?  
        and many more and I'll do my best to answer them!    
            ''')
btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])
