import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import dotenv
from typing import List, Optional, Dict, Any
from collections import defaultdict
import uuid
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from googlesearch import search
import requests
import time
from datetime import datetime

# Load environment variables from .env file
dotenv.load_dotenv()

# Set up page configuration
st.set_page_config(
    page_title="Interactive Knowledge Base Chat",
    page_icon="💬",
    layout="wide"
)

dork = "site:ttu.edu"




# Define Pydantic model for structured output
class RAGResponse(BaseModel):
    result: str = Field(description="The answer to the user's question based on the context")
    retry_needed: bool = Field(description="Flag indicating if additional information retrieval is needed (True/False)")
    retry_question: Optional[str] = Field(None, description="Specific question to retrieve additional context if retry_needed is True")
    missing_information: Optional[str] = Field(None, description="Description of what information is missing from the context")
    sources_used: List[str] = Field([], description="List of source chunks used to answer the question")
    question_to_user: Optional[str] = Field(None, description="Question to ask the user if more information is needed")

# Define Chat Message model
class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    metadata: Dict[str, Any] = Field(default_factory=dict)  # For storing sources, retry info, etc.

# Path to your markdown files
MARKDOWN_DIR = "./markdown_output/"  # Change this to your actual directory
DB_DIR = "vectorstore_db"  # Directory to store vector database
MAX_RETRIES = 3  # Maximum number of retry attempts
INITIAL_CHUNK_COUNT = 8  # Start with more chunks to improve topic coverage
RETRY_CHUNK_COUNT = 5  # Number of chunks to retrieve on each retry
MAX_CONTEXT_MESSAGES = 10  # Maximum number of previous messages to include in context
DEEP_RESEARCH = False  # Enable deep research mode
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 1000
BATCH_SIZE = 145  # Adjust as needed
SLEEP_TIME = 20  # Time (seconds) to sleep between batches


# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = {}

# Function to initialize the vector database
@st.cache_resource
def initialize_db():
    """Load documents, create chunks, generate embeddings, and create vector store."""
    try:
        # Check if vector DB already exists
        if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
            info = st.info("Please wait while loading the existing vector database...")
            time.sleep(2)  # Pause for 2 seconds to display info message
            info.empty()  # Clear info message
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
            return db

        # If not, create a new one
        st.info("Creating new vector database from markdown files...")

        # Load all markdown files from directory
        loader = DirectoryLoader(
            MARKDOWN_DIR,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={'autodetect_encoding': True}
        )
        documents = loader.load()

        st.info(f"Loaded {len(documents)} markdown files")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        st.info(f"Split into {len(chunks)} chunks")

        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        # Function to split list into batches
        def split_list(input_list, chunk_size):
            for i in range(0, len(input_list), chunk_size):
                yield input_list[i:i + chunk_size]

        # Process chunks in batches
        split_chunks = split_list(chunks, BATCH_SIZE)

        for i, chunk_batch in enumerate(split_chunks):
            print(f"Processing batch {i+1}...")

            try:
                vectordb = Chroma.from_documents(
                    documents=chunk_batch,
                    embedding=embeddings,  # Fixed variable name
                    persist_directory=DB_DIR
                )
                # vectordb.persist()

                print(f"Batch {i+1} processed successfully. Sleeping for {SLEEP_TIME} seconds...")
                time.sleep(SLEEP_TIME)  # Wait to avoid rate limits

            except Exception as e:
                print(f"Error in batch {i+1}: {e}")
                continue  # Continue processing next batch even if one fails

        st.success(f"Vector database created with {len(chunks)} text chunks")
        time.sleep(2)  # Pause for 2 seconds to display success message
        st.empty()  # Clear success message
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    except Exception as e:
        st.error(f"Error initializing database: {str(e)}")
        return None

# Function to retrieve documents from vector store
def retrieve_documents(query, db, k=5, filter_sources=None):
    """
    Retrieve relevant documents from the vector store.

    Args:
        query: The search query
        db: The vector database
        k: Number of documents to retrieve
        filter_sources: Optional list of sources to exclude

    Returns:
        List of retrieved documents
    """
    retriever = db.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)

    # If we have sources to filter out, do so
    if filter_sources:
        filtered_docs = []
        for doc in docs:
            source = getattr(doc, 'metadata', {}).get('source', 'Unknown')
            # Only keep documents whose sources are not in the filter list
            if source not in filter_sources:
                filtered_docs.append(doc)
        return filtered_docs

    return docs

# Function to expand document retrieval to find related chunks
def retrieve_with_expansion(initial_query, db, initial_k=8, expansion_k=3):
    """
    Retrieve documents and then expand with additional retrievals to catch all related chunks.

    This helps when information about a topic is split across multiple chunks.
    """
    # Initial document retrieval
    initial_docs = retrieve_documents(initial_query, db, k=initial_k)

    # Track all retrieved documents and their sources
    all_docs = list(initial_docs)
    retrieved_sources = {getattr(doc, 'metadata', {}).get('source', 'Unknown') for doc in all_docs}

    # For each document, try to find nearby chunks from the same source
    # This helps when information is split across consecutive chunks in the same file
    for doc in initial_docs:
        source = getattr(doc, 'metadata', {}).get('source', 'Unknown')
        if source == 'Unknown':
            continue

        # Extract content snippets to use as expansion queries
        content = doc.page_content
        # Use the first 100 characters as a query to find adjacent chunks
        snippet_query = content[:min(100, len(content))]

        # Retrieve additional chunks potentially from the same document
        expansion_docs = retrieve_documents(snippet_query, db, k=expansion_k)

        # Add new documents that weren't in the initial set
        for exp_doc in expansion_docs:
            exp_source = getattr(exp_doc, 'metadata', {}).get('source', 'Unknown')
            # Only add if it's from the same source but not the exact same content
            if exp_source == source and exp_doc.page_content != doc.page_content:
                all_docs.append(exp_doc)

    return all_docs

# Function to build conversation context from chat history
def build_conversation_context(current_query):
    """
    Build a context string from recent chat history for context-aware responses.
    """
    if not st.session_state.chat_history:
        return current_query

    # Get the most recent messages, limited to MAX_CONTEXT_MESSAGES
    recent_messages = st.session_state.chat_history[-MAX_CONTEXT_MESSAGES:]

    # Format the conversation context
    context_parts = []

    # Add previous turns
    for msg in recent_messages:
        prefix = "User: " if msg.role == "user" else "Assistant: "
        context_parts.append(f"{prefix}{msg.content}")

    # Add the current query
    context_parts.append(f"User: {current_query}")

    # Join everything
    conversation_context = "\n\n".join(context_parts)

    return conversation_context

# Function to get response from LLM with retry logic and conversation context
def get_response_with_context(query, db):
    """Get response for the query using RAG with retry logic, context awareness, and comprehensive topic retrieval."""
    # Initialize the parser
    parser = PydanticOutputParser(pydantic_object=RAGResponse)

    TODAY = datetime.today().strftime('%Y-%m-%d')


    # Build conversation context from chat history
    conversation_context = build_conversation_context(query)

    # Search for additional context if deep research is enabled

    # Create a template for context-aware prompting with structured output
    template = """
    You are a knowledgeable assistant that provides accurate information based on the given context while maintaining conversation continuity.

    Previous conversation:
    {conversation_history}

    Document context information:
    {document_context}

    Current User Question: {question}

    Please analyze the available context, previous conversation, and provide a response according to these guidelines:
    1. Consider both the document context AND the conversation history when forming your answer.
    2. If the user is asking a follow-up question, use the conversation history to understand what they're referring to.
    3. If the context contains sufficient information to answer the question thoroughly, provide a comprehensive answer.
    4. If the context lacks important information to answer the question properly, indicate that a retry is needed by setting retry_needed=True.
    5. If retry is needed, specify what additional information to search for in the retry_question field.
    6. Describe what information is missing in the missing_information field.
    7. Always list the sources you used in your response in the sources_used field, but DO NOT mention the sources directly in your result text.
    8. Make your response conversational and fluid, acknowledging previous exchanges when appropriate.
    9. Look for repeated information and avoid redundancy.
    10. Provide a clear and concise response that directly addresses the user's query.
    11. If the context is insufficient, ask clarifying questions to user back to gather more information.
    12.If Search context is provided, use it to get more information and mention in response that you have used search context to get more information else ignore it.
    13 Todays date is {TODAY}, Use the date only if the context is related to the current date(when something about deadlines or date realted) else ignore it.

    {format_instructions}
    """

    # Add format instructions to the prompt
    format_instructions = parser.get_format_instructions()
    prompt = PromptTemplate(
        template=template,
        input_variables=["document_context", "conversation_history", "question", "TODAY"],
        partial_variables={"format_instructions": format_instructions}
    )

    # Create the LLM
    # llm = ChatOpenAI(
    #     model="gpt-4o",
    #     temperature=0.1
    # )
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,)

    # Initialize retry counter and sources used
    retry_count = 0
    current_question = query

    # Store all retrieved documents and their metadata
    all_docs = []
    all_sources = set()  # Use a set to avoid duplicate sources

    # Create a placeholder for the response
    response_placeholder = st.empty()

    # Initialize progress tracking
    progress = st.progress(0)

    while retry_count < MAX_RETRIES:
        # Update progress
        progress_value = retry_count / MAX_RETRIES
        progress.progress(progress_value)

        # Display current retrieval status
        retry_info = f"Retrieval attempt {retry_count + 1}/{MAX_RETRIES}" if retry_count > 0 else "Initial retrieval"
        response_placeholder.info(f"{retry_info}: Searching for relevant information about '{current_question}'")

        # Use expanded retrieval for first attempt, standard for retries
        if retry_count == 0:
            # First retrieval - use expanded retrieval to capture related chunks
            new_docs = retrieve_with_expansion(current_question, db,
                                              initial_k=INITIAL_CHUNK_COUNT,
                                              expansion_k=3)
        else:
            # Subsequent retrievals - standard retrieval but exclude previously used sources
            new_docs = retrieve_documents(current_question, db, k=RETRY_CHUNK_COUNT)

        # Add new documents to the accumulated list
        all_docs.extend(new_docs)

        # Extract all document contents and sources
        all_contexts = [doc.page_content for doc in all_docs]
        doc_sources = [getattr(doc, 'metadata', {}).get('source', 'Unknown source') for doc in all_docs]

        # Update all_sources
        all_sources.update(doc_sources)

        # Join all contexts with separators and add source information
        combined_contexts = []
        for i, (content, source) in enumerate(zip(all_contexts, doc_sources)):
            # Add a header for each chunk that includes its source
            source_name = os.path.basename(source) if isinstance(source, str) else "Unknown source"
            combined_contexts.append(f"--- CHUNK {i+1} (FROM: {source_name}) ---\n{content}")

        document_context = "\n\n".join(combined_contexts)

        try:
            # Generate response with both document context and conversation history
            response_text = llm.invoke(
                prompt.format(
                    document_context=document_context,
                    conversation_history=conversation_context,
                    question=query,
                    TODAY=TODAY
                )
            ).content

            # Parse the response
            parsed_response = parser.parse(response_text)

            # Add sources to response (as a list from the set)
            parsed_response.sources_used = list(all_sources)

            # Display retry information if needed
            if parsed_response.retry_needed and retry_count < MAX_RETRIES - 1:
                retry_count += 1
                current_question = parsed_response.retry_question

                # Update progress
                response_placeholder.info(
                    f"Retry {retry_count}/{MAX_RETRIES}: More information needed. "
                    f"Searching for: '{parsed_response.retry_question}'"
                )
            else:
                # Clear the placeholder before returning
                response_placeholder.empty()
                progress.progress(1.0)  # Set progress to 100%
                return parsed_response

        except Exception as e:
            # Clear the placeholder
            response_placeholder.empty()
            progress.progress(1.0)  # Set progress to 100%
            st.error(f"Error processing response: {str(e)}")

            # Create a fallback response
            return RAGResponse(
                result=f"An error occurred while processing your query: {str(e)}",
                retry_needed=False,
                retry_question=None,
                missing_information=None,
                sources_used=list(all_sources)
            )

    # Clear the placeholder in case we exit the loop
    response_placeholder.empty()
    progress.progress(1.0)  # Set progress to 100%

    # If we've exhausted retries, return the last response
    return parsed_response

# Function to add a message to chat history
def add_message_to_history(role, content, metadata=None):
    """Add a new message to the chat history."""
    if metadata is None:
        metadata = {}

    message = ChatMessage(
        role=role,
        content=content,
        metadata=metadata
    )

    st.session_state.chat_history.append(message)

# Function to display chat messages
def display_chat_messages():
    """Display all messages in the chat history."""
    for message in st.session_state.chat_history:
        with st.chat_message(message.role):
            st.markdown(message.content)

            # If it's an assistant message with sources, add an expander
            if message.role == "assistant" and "sources_used" in message.metadata and message.metadata["sources_used"]:
                with st.expander("Sources", expanded=False):
                    # Group sources by filename
                    sources_by_file = defaultdict(int)
                    for source in message.metadata["sources_used"]:
                        source_name = os.path.basename(source) if isinstance(source, str) else "Unknown source"
                        sources_by_file[source_name] += 1

                    # Display sources sorted by count (most relevant first)
                    for filename, count in sorted(sources_by_file.items(), key=lambda x: x[1], reverse=True):
                        st.markdown(f"• {filename} ({count} chunks)")

# Function to handle user messages
def handle_user_message(user_query):
    """Process user message, get a response, and update chat history."""
    # Add user message to history
    add_message_to_history("user", user_query)

    # Display "Thinking..." while processing
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("Thinking...")

        try:
            # Get response
            response = get_response_with_context(user_query, db)

            # Create metadata for assistant message
            metadata = {
                "sources_used": response.sources_used,
                "retry_needed": response.retry_needed,
                "retry_question": response.retry_question,
                "missing_information": response.missing_information
            }

            # Add assistant message to history
            add_message_to_history("assistant", response.result, metadata)

            # Update the thinking placeholder with the actual response
            thinking_placeholder.empty()
            st.markdown(response.result)

            # If sources exist, add them to an expander
            if response.sources_used:
                with st.expander("Sources", expanded=False):
                    # Group sources by filename
                    sources_by_file = defaultdict(int)
                    for source in response.sources_used:
                        source_name = os.path.basename(source) if isinstance(source, str) else "Unknown source"
                        sources_by_file[source_name] += 1

                    # Display sources sorted by count
                    for filename, count in sorted(sources_by_file.items(), key=lambda x: x[1], reverse=True):
                        st.markdown(f"• {filename} ({count} chunks)")

        except Exception as e:
            # Handle errors
            error_message = f"I encountered an error: {str(e)}"
            thinking_placeholder.empty()
            st.markdown(error_message)

            # Add error message to history
            add_message_to_history("assistant", error_message)

# Streamlit UI setup
st.title("💬 Interactive Knowledge Base Chat")
st.subheader("Ask questions about your markdown documents with conversation memory")

# Sidebar - API Key and configuration
with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input("Enter your GEMINI API Key", type="password")
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key

    # Add configuration options
    st.subheader("Retrieval Settings")
    max_retries = st.slider("Maximum Retries", min_value=0, max_value=5, value=MAX_RETRIES)
    initial_chunks = st.slider("Initial Chunks to Retrieve", min_value=3, max_value=15, value=INITIAL_CHUNK_COUNT)
    retry_chunks = st.slider("Chunks per Retry", min_value=2, max_value=10, value=RETRY_CHUNK_COUNT)
    context_messages = st.slider("Max Context Messages", min_value=1, max_value=20, value=MAX_CONTEXT_MESSAGES)
    deep_research = st.checkbox("Enable Deep Research", value=False)

    # Update global variables with slider values
    MAX_RETRIES = max_retries
    INITIAL_CHUNK_COUNT = initial_chunks
    RETRY_CHUNK_COUNT = retry_chunks
    MAX_CONTEXT_MESSAGES = context_messages

    # Add a clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.conversation_context = {}
        st.rerun()

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application uses:
    - LangChain for document processing
    - OpenAI Embeddings for vector encoding
    - GPT-4o with structured output for answering questions
    - Chroma as the vector database
    - Conversation memory for follow-up questions
    - Comprehensive topic retrieval
    - Smart retry mechanism
    """)

# Initialize vector DB only if API key is provided
db = None
if os.environ.get("GEMINI_API_KEY"):
    db = initialize_db()

    # If database is successfully initialized
    if db:
        # Display existing chat messages
        display_chat_messages()

        # Chat input
        user_query = st.chat_input("Ask a question about your markdown files")

        # Process user input when submitted
        if user_query:
            handle_user_message(user_query)
else:
    st.warning("Please enter your GEMINI API key in the sidebar to get started.")

# Add a reset button to rebuild the database
if db and st.sidebar.button("Rebuild Database"):
    import shutil
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    st.cache_resource.clear()
    st.rerun()