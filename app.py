from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

from langchain.agents import initialize_agent
from langchain.agents import AgentType
import os
import databutton as db
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from pydantic.v1 import BaseModel, Field
import pandas as pd
import datetime
from streamlit_feedback import streamlit_feedback
import time


st.title("Your AI-Powered Digital Marketing Assistant")
st.header("PDF Comparison Chatbot with User Insights")
st.markdown('''

> Built using: `LangChain AI`  `Databutton` `Trubrics`

'''

)

st.markdown('''
        [Medium Blog Post](https://medium.com/@avra42/building-an-ai-powered-digital-marketing-assistant-acfd302554f0)
        '''
        )

@st.cache_resource
def perform_embeddings_and_prepare_tools():
    class DocumentInput(BaseModel):
        question: str = Field()
    # st.toast("Performing embeddings.")
    LinkedIn = db.storage.binary.get(key="mock-data-komplett-linked-in-pdf")
    Twitter = db.storage.binary.get(key="mock-data-komplett-twitter-pdf")

    llm = ChatOpenAI(
        temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=OPENAI_API_KEY
    )

    tools = []
    files = [{"name": "LinkedIn", "rename": "LinkedIn Report"}, {"name": "Twitter", "rename": "Twitter Report"}]
    for file in files:
        import tempfile

        with tempfile.NamedTemporaryFile("wb") as f:
            if file["name"] == "LinkedIn":
                write_file = f.write(LinkedIn)
            else:
                write_file = f.write(Twitter)
            
            temp_filepath = f.name
            # file_on_expander = file["rename"]
            loader = PyPDFLoader(temp_filepath)
            pages = loader.load_and_split()
            # with st.expander(f"{file_on_expander}"):
            #     pages

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(pages)
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            retriever = FAISS.from_documents(docs, embeddings).as_retriever()

                # Wrap retrievers in a Tool
            tools.append(
                Tool(
                    args_schema=DocumentInput,
                    name=file["name"],
                    description=f"useful when you want to answer questions about {file['name']}",
                    func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
                )
            )
    st.info("Embeddings are performed and cached for a single session. You can start chatting using the input box below 💬", icon = "✅")
    return tools

st.subheader("Introduction")

st.write('''

        **Description:** This app's primary purpose is to analyze multiple reports while incorporating a chatbot for user interaction. 
        It aims to collect user feedback to assess the app's utility, ultimately striving to create an exceptionally valuable tool
        for your internal teams or external clients.
        
        **Details:** It takes two simulated social media reports (LinkedIn & Twitter) from the biggest e-commerce 
        player in the Nordics, Komplett Group—then uses LangChain agents to compare them. After every successful 
        chat interaction, users can give their feedback, and this feedback is instantly stored in Databutton's storage system. 
        The chatbot's feedback tool relies on the Streamlit-feedback Python package from Trubrics. 
        
        **Note:** Please keep in mind that this data is entirely fictional and for illustrative purposes only. 
        Actual Komplett social media metrics are unknown to us.

''')

with st.chat_message("Assistant"):
    mp = st.empty()
    sl = mp.container()
    sl.markdown( """
        Hi there! I'm your Digital Marketing Assistant! 
        
        **Getting started:** 
        1. I need your OpenAI API key to work. If you don't have a key, you can sign-up and create one here 
        https://platform.openai.com/account/api-keys. Don't worry, your key will not be stored in Databutton.
        2. Use the "Example questions" tab for pre-tested chatbot prompts.
    """
    )

# OPENAI_API_KEY = sl.text_input("Type in your OpenAI API key to continue", type="password")

# st.session_state.api = OPENAI_API_KEY
OPENAI_API_KEY = db.secrets.get(name="OPENAI_API_KEY")

if OPENAI_API_KEY:

    st.subheader("Example Question")
    st.code("Which has a better follower growth rate in 2023, LinkedIn or Twitter?")
    mp = st.empty()

        # st.toast("Preparing Tools to serve ⚒️ ")
    tools = perform_embeddings_and_prepare_tools()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    
    llm = ChatOpenAI(
        temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=OPENAI_API_KEY
    )
    
    agent = initialize_agent(
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=tools,
        llm=llm,
        verbose=True,
    )

    

    if prompt := st.chat_input(placeholder="Which has a better follower growth rate in 2023, LinkedIn or Twitter?"):
        st.chat_message("user").write(prompt)
    
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
    
        with st.chat_message("assistant"):
            
            full_response = ""
            st_callback = StreamlitCallbackHandler(st.container())
            message_placeholder = st.empty()
            response = agent.run(prompt, callbacks=[st_callback])
            
            # Simulate stream of response with milliseconds delay
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
        # Feedback

    if st.session_state.messages and len(st.session_state.messages) % 2 == 0:
        feedback = streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="[Optional] Please provide an explanation",
        )
        
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        
           
        if feedback:
            existing_df = db.storage.dataframes.get(
                key="response", default=lambda: pd.DataFrame()
            )
        
            # Extract the desired fields
            score = feedback["score"]
            text = feedback["text"]
            created_on = current_time
        
            df = pd.DataFrame({"score": [score], "text": [text], "created_on": [created_on]})
            df_to_store = pd.concat([existing_df, df], ignore_index=True)
            db.storage.dataframes.put(key="response", value=df_to_store)
            st.toast(" Thanks! Your feedback is updated to the database.")
            
        
