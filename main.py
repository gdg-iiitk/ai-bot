from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime

# Set your API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAYew4okjx4jmR7xbKhLj2mAckgtUUbR-k"

# Initialize the LangChain Google Gemini model
llm = ChatGoogleGenerativeAI(
    # model="gemini-1.5-pro",
    model="gemini-2.0-flash-exp",
    temperature=1.0,
    top_p=0.95,
    top_k=40,
    max_output_tokens=8192,
)

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
with open("mess_menu.txt", "r") as file:
    mess_menu = file.read()
def load_mess_menu(input_str=None):
    """Load the mess menu context into the conversation"""
    return f"This is the mess menu of the campus. Please use this context to answer queries about the menu: {mess_menu}"


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
        # return_direct=True  # Add this to return the result directly
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

def chat(user_input):
    # Get the response from the agent
    response = agent.run(user_input)
    
    # Update the conversation memory with the response
    memory.save_context({"input": user_input}, {"output": response})
    
    return response

# Example usage
user_input = input("Enter your query: ")
response = chat(user_input)
print("Assistant:", response)
