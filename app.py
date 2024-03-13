import os
import streamlit as st
import arxiv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.callbacks.base import BaseCallbackHandler
import tiktoken
from dotenv import load_dotenv
from hashlib import sha256
import time

# Load environment variables
load_dotenv()

# Load the password from the .env file
stored_password = os.getenv("APP_PASSWORD")

# Initialize environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "SDSC6002"

# Initialize session state variables
if 'max_docs' not in st.session_state:
    st.session_state.max_docs = 10
if 'sort_method' not in st.session_state:
    st.session_state.sort_method = "Relevance"
if 'search_result' not in st.session_state:
    st.session_state.search_result = []
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=f"***Document Summary***\n\n", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")

# Set up Streamlit layout and interface elements
st.set_page_config(
    page_title="ArXiv Document Search and Summarization",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'About': "This is an app for searching and summarizing ArXiv documents."
    }
)

if st.session_state['authenticated']:

    # Sidebar settings
    st.sidebar.header('Settings')
    st.session_state.max_docs = st.sidebar.slider("Maximum Number of Documents to Search", 1, 20, 10)
    st.session_state.sort_method = st.sidebar.radio("Sorting Method of the Search", ["Relevance", "Submission Date"])

    # Main page for search
    st.markdown("""
        <h1 style='font-size: 24px;'>ArXiv Document Search and Summarization</h1>
    """, unsafe_allow_html=True)
    search_keywords = st.text_input('Input keywords for document search (e.g. artificial intelligence)', '')

    def ArXiv_Search(keywords):
        sort_criteria = arxiv.SortCriterion.SubmittedDate if st.session_state.sort_method == "Submission Date" else arxiv.SortCriterion.Relevance
        search = arxiv.Search(query=keywords, max_results=st.session_state.max_docs, sort_by=sort_criteria)
        result_list = []
        for result in search.results():
            result_dict = {
                "id": result.get_short_id(),
                "title": result.title,
                "date_published": result.published,
                "summary": result.summary,
                "article_url": result.entry_id,
                "pdf_url": result.pdf_url
            }
            result_list.append(result_dict)
        st.session_state.search_result = result_list

    if st.button('Search'):
        ArXiv_Search(search_keywords)

    # Display search results and selection
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.session_state.search_result:
            titles = [doc['title'] for doc in st.session_state.search_result]
            selected_title = st.radio("Select a document to summarize", titles, index=0)

    def download_and_summarize_document(selected_doc):
        # Placeholder for the progress bar
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)


        with st.spinner('Downloading and processing the document...'):
                                   
            # Simulate progress
            progress_bar.progress(10)

            # Create directory if it doesn't exist
            directory = './pdf_docs'
            Path(directory).mkdir(parents=True, exist_ok=True)

            # Download PDF
            target_id = selected_doc['id']
            paper = next(arxiv.Client().results(arxiv.Search(id_list=[target_id])))
            pdf_path = paper.download_pdf(directory)
            
            # Update progress after downloading
            progress_bar.progress(30)

            # Load PDF
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            # Combine documents
            pdf_text = " ".join(doc.page_content for doc in docs)
            combined_doc = [Document(page_content=pdf_text)]
            
            # Tokenize and check length
            tokenizer = tiktoken.get_encoding("cl100k_base")
            tokens = tokenizer.encode(pdf_text)
            
            # Update progress after downloading
            progress_bar.progress(50)        

            # Summarize based on token length

        with st.spinner('Summarizing document...'):

            # Placeholder for Output Box
            output_placeholder = st.empty()
            
            stream_handler = StreamHandler(output_placeholder, display_method='write')
            
            if len(tokens) <= 120000:
                # Use Stuff Method for summarization
                
                # Define prompt
                prompt_template = """Write a concise summary of the following:
                "{text}"
                The summary should be structured in bulleted lists following the headings Core Argument, Evidence, and Conclusions.
                CONCISE SUMMARY:"""
                prompt = PromptTemplate.from_template(prompt_template)

                # Define LLM 
                
                # Option 1: Azure
                # llm = AzureChatOpenAI(
                #     openai_api_version="2023-05-15",
                #     azure_deployment="gpt-4-turbo-128K",
                #     temperature=0,
                #     seed=11,
                #     streaming=True,
                #     callbacks=[stream_handler]
                # )
                
                # Option 2: OpenAI
                llm = ChatOpenAI(
                    model="gpt-4-turbo-preview",
                    temperature=0,
                    seed=11,
                    streaming=True,
                    callbacks=[stream_handler]
                )

                # Define LLM chain
                llm_chain = LLMChain(llm=llm, prompt=prompt)

                # Define StuffDocumentsChain
                stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
                
                response = stuff_chain.invoke(combined_doc)
            else:
                # Use Map-Reduce Method for summarization
                
                # Define LLM #1
                # Option 1: Azure
                # llm1 = AzureChatOpenAI(
                #     openai_api_version="2023-05-15",
                #     azure_deployment="gpt-4-turbo-128K",
                #     temperature=0,
                #     seed=11
                # )                
                
                # Option 2: OpenAI
                llm1 = ChatOpenAI(
                    model="gpt-4-turbo-preview",
                    temperature=0,
                    seed=11
                )

                # Define LLM #2
                # Option 1: Azure
                # llm2 = AzureChatOpenAI(
                #     openai_api_version="2023-05-15",
                #     azure_deployment="gpt-4-turbo-128K",
                #     temperature=0,
                #     seed=11,
                #     streaming=True,
                #     callbacks=[stream_handler]
                # )

                # Option 2: OpenAI
                llm2 = ChatOpenAI(
                    model="gpt-4-turbo-preview",
                    temperature=0,
                    seed=11,
                    streaming=True,
                    callbacks=[stream_handler]
                )                
             
                # Map
                map_template = """The following is the content of a particular section of the document:
                {docs}
                Based on the content, please summarize key contents.
                Helpful Answer:"""
                map_prompt = PromptTemplate.from_template(map_template)
                map_chain = LLMChain(llm=llm1, prompt=map_prompt)
                
                # Reduce
                reduce_template = """The following is set of summaries:
                {docs}
                Take these and distill it into a final, consolidated summary of the main themes.
                The summary should be structured in bulleted lists following the headings Core Argument, Evidence, and Conclusions.
                Helpful Answer:"""
                reduce_prompt = PromptTemplate.from_template(reduce_template)

                # Run chain
                reduce_chain = LLMChain(llm=llm2, prompt=reduce_prompt)

                # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
                combine_documents_chain = StuffDocumentsChain(
                    llm_chain=reduce_chain, document_variable_name="docs"
                )

                # Combines and iteratively reduces the mapped documents
                reduce_documents_chain = ReduceDocumentsChain(
                    # This is final chain that is called.
                    combine_documents_chain=combine_documents_chain,
                    # If documents exceed context for `StuffDocumentsChain`
                    collapse_documents_chain=combine_documents_chain,
                    # The maximum number of tokens to group documents into.
                    token_max=120000,
                )

                # Combining documents by mapping a chain over them, then combining results
                map_reduce_chain = MapReduceDocumentsChain(
                    # Map chain
                    llm_chain=map_chain,
                    # Reduce chain
                    reduce_documents_chain=reduce_documents_chain,
                    # The variable name in the llm_chain to put the documents in
                    document_variable_name="docs",
                    # Return the results of the map steps in the output
                    return_intermediate_steps=False,
                )

                text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=120000, chunk_overlap=1000
                )
                split_docs = text_splitter.split_documents(combined_doc)
                
                response = map_reduce_chain.invoke(split_docs)
            
            # Complete the progress
            progress_bar.progress(100)

            # Clear the progress bar after completion
            progress_placeholder.empty()           

    with col2:
        if st.session_state.search_result and selected_title:
            selected_doc = next((doc for doc in st.session_state.search_result if doc['title'] == selected_title), None)
            if selected_doc:
                # Display document information
                st.markdown(f"**ID:** {selected_doc['id']}")
                st.markdown(f"**Title:** {selected_doc['title']}")
                st.markdown(f"**Published:** {selected_doc['date_published']}")
                st.markdown(f"**Abstract:** {selected_doc['summary']}")
                st.markdown(f"**Article URL:** [Link]({selected_doc['article_url']})")
                st.markdown(f"**PDF URL:** [Link]({selected_doc['pdf_url']})", unsafe_allow_html=True)
                
                # Add Summarize button and handle summarization
                if st.button('Summarize'):
                    download_and_summarize_document(selected_doc)
else:
    st.title("Welcome to ArXiv Document Search and Summarization Engine")

    entered_password = st.text_input("Please enter password", type="password")
    submit_password = st.button("Submit")

    if submit_password:
        # Hash the entered password
        hashed_entered_password = sha256(entered_password.encode('utf-8')).hexdigest()

        if hashed_entered_password == stored_password:
            st.success("Access granted.")
            st.session_state['authenticated'] = True
            # Refresh the page after a short delay
            time.sleep(1)
            st.experimental_rerun()
        else:
            st.error("Wrong Password.  Please retry!")
