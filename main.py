from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.vectorstores import Chroma
from langchain.chains import ConversationChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
import os
from datetime import datetime

# Set your API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAYew4okjx4jmR7xbKhLj2mAckgtUUbR-k"

llm = ChatGoogleGenerativeAI(
    # model="gemini-1.5-pro",
    model="gemini-2.0-flash-exp",
    temperature=1.0,
    top_p=0.95,
    top_k=40,
    max_output_tokens=8192,
)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize ChromaDB
PERSIST_DIRECTORY = "db"
vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

def add_document(file_path):
    """Add a document to the vector database"""
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    vectorstore.add_documents(texts)
    vectorstore.persist()
    return f"Added {len(texts)} chunks from {file_path}"

def add_directory(dir_path):
    """Add all text files from a directory to the vector database"""
    loader = DirectoryLoader(dir_path, glob="**/*.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    vectorstore.add_documents(texts)
    vectorstore.persist()
    return f"Added {len(texts)} chunks from directory {dir_path}"

def add_text(text, metadata=None):
    """Add raw text directly to the vector database"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t, metadata=metadata or {}) for t in texts]
    vectorstore.add_documents(docs)
    vectorstore.persist()
    return f"Added {len(texts)} chunks of text"

def search_knowledge_base(query, k=3):
    """Search the vector database for relevant context"""
    docs = vectorstore.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])

def clear_database():
    """Clear all documents from the vector database"""
    vectorstore.delete_collection()
    vectorstore.persist()
    return "Database cleared"

# Create a conversation memory
memory = ConversationBufferMemory()

# Define the conversation template
template = """You are an AI assistant for IIIT Kottayam students.
Current conversation:
{history}
Human: {input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)

# Create the conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)


# Define tools
with open("./data/premade/mess_menu.txt", "r") as file:
    mess_menu = file.read()
with open("./data/premade/inst_ calender.txt", "r") as file:
    inst_calendar = file.read()

def load_mess_menu(input_str=None):
    """Load the mess menu context into the conversation"""
    return f"This is the mess menu of the campus. Please use this context to answer queries about the menu: {mess_menu}"

def load_calendar(input_str=None):
    """Load the institute calendar context into the conversation"""
    return f"This is the academic calendar of IIIT Kottayam. Please use this context to answer queries about academic dates and holidays: {inst_calendar}"

def get_time_context(arg):
    """Get current date and time information"""
    print(arg)
    now = datetime.now()
    return f"""Current time context:
- Date: {now.strftime('%A, %B %d, %Y')}
- Time: {now.strftime('%I:%M %p')}
- Day of week: {now.strftime('%A')}"""

tools = [
    Tool(
        name="Load Mess Menu",
        func=load_mess_menu,
        description="Use this tool to load the mess menu context when the query is related to food, mess, or menu.",
    ),
    Tool(
        name="Load Academic Calendar",
        func=load_calendar,
        description="Use this tool to load the academic calendar when the query is related to academic dates, holidays, or institute events like exams.",
    ),
    Tool(
        name="Get Date and Time Context",
        func=get_time_context,
        description="Use this tool to get the current date and time information and what day it is"
    )
]

# Initialize agent
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    allow_multiple_tools=True  # Enable multiple tools
)

# Replace the old conversation chain with a retrieval chain
qa_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
    memory=qa_memory,
    verbose=True
)

# Modify chat function to handle errors
def chat(user_input):
    try:
        # First try to get relevant context-based response
        qa_response = qa_chain({"question": user_input})
        context_response = qa_response["answer"]
        
        # Then try to use tools if needed
        tool_response = agent.run(user_input)
        
        # Combine responses intelligently
        final_response = f"{context_response}\n\n{tool_response}" if tool_response else context_response
        
        return final_response
    except Exception as e:
        if "429" in str(e):
            return "Sorry, I'm a bit busy right now. Please try again in a moment."
        return f"I encountered an error: {str(e)}"

def initialize_bot():
    """Initialize the chatbot and return the chat function"""
    if not os.path.exists(PERSIST_DIRECTORY):
        add_document("./data/premade/mess_menu.txt")
        add_document("./data/premade/inst_calender.txt")
    
    return chat

if __name__ == "__main__":
    initialize_bot()
    while True:
        user_input = input("Enter your query: ")
        response = chat(user_input)
        print("Assistant:", response)
