from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
import os
import logging
from datetime import datetime
from vdb_management import VectorStoreManager

# Set up API key and logging
os.environ["GOOGLE_API_KEY"] = "AIzaSyAYew4okjx4jmR7xbKhLj2mAckgtUUbR-k"

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    # model="gemini-1.5-pro",
    model="gemini-2.0-flash-exp",
    temperature=1.0,
    top_p=0.95,
    top_k=40,
    max_output_tokens=8192,
)

logging.basicConfig(filename='chatbot.log', level=logging.INFO)


# Initialize vector store manager class:-
vector_manager = VectorStoreManager(persist_directory="db")

# Load context files

# Define tools with context loading
tools = [
    Tool(
        name="Load Mess Menu",
        func=lambda x: f"Mess menu context: {load_context_file('mess_menu.txt')}",
        description="Load mess menu context for food-related queries",
    ),
    Tool(
        name="Load Academic Calendar",
        func=lambda x: f"Academic calendar context: {load_context_file('inst_calender.txt')}",
        description="Load academic calendar context for academic dates and holidays",
    ),
    Tool(
        name="Get Date and Time Context",
        func=lambda x: f"Current time context: {datetime.now().strftime('%A, %B %d, %Y %I:%M %p')}",
        description="Get current date and time information",
    ),
    Tool(
        name="Load Curriculum",
        func=lambda x: f"Curriculum context: {load_context_file('caricululm.txt')}",
        description="Load curriculum context for course-related queries",
    ),
    Tool(
        name="Load Milma Menu",
        func=lambda x: f"Milma Cafe menu context: {load_context_file('milma_menu.txt')}",
        description="Load Milma Cafe menu context for food-related queries",
    ),
    Tool(
        name="Load Faculty Details",
        func=lambda x: f"Faculty details context: {load_context_file('faculty_details.txt')}",
        description="Load faculty details context for faculty-related queries",
    )
]

def load_context_file(filename):
    try:
        with open(f"./data/{filename}", "r") as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error loading {filename}: {e}")
        return ""
    
# Initialize agent
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    allow_multiple_tools=True
)

# Define conversation template
template = """You are Mr.G, an AI assistant for IIIT Kottayam students.
Current conversation:
{chat_history}
Context: {context}
Human: {question}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"], 
    template=template
)

# Initialize retrieval components
retrieval_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Create retrieval chain
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_manager.get_retriever(),
    memory=retrieval_memory,
    get_chat_history=lambda h: h,
    combine_docs_chain_kwargs={"prompt": prompt, "document_variable_name": "context"},
    verbose=True
)

def chat(user_input):
    try:
        logging.info(f"User Input: {user_input}")
        
        retrieval_response = retrieval_chain({"question": user_input})
        context_response = retrieval_response["answer"]
        tool_response = agent.run(user_input)
        
        responses = [r.strip() for r in [context_response, tool_response] if r and r.strip()]
        final_response = "\n\n".join(responses)
        
        return {
            "final_response": final_response.split("Final Answer:")[-1].strip() if "Final Answer:" in final_response else final_response,
            "full_log": {
                "context_response": context_response,
                "tool_response": tool_response,
                "final_response": final_response
            }
        }
    except Exception as e:
        logging.error(f"Error in chat: {str(e)}")
        return {"final_response": f"I encountered an error: {str(e)}"}

def initialize_bot():
    """Initialize the chatbot and return the chat function"""
    vector_manager.initialize_from_directory("./data")
    return chat

if __name__ == "__main__":
    initialize_bot()
    while True:
        user_input = input("\nEnter your query (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        response = chat(user_input)
        print("Assistant:", response["final_response"])