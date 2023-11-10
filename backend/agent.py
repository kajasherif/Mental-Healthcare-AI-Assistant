from langchain.chat_models import AzureChatOpenAI
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from prompt import PREFIX, FORMAT_INSTRUCTIONS, SUFFIX
from custom_tools import tools_diff  # Make sure this import is correct
from utils import global_dict, global_unique
from chains import TherapyStageAnalyzerChain, TherapyConversationChain, TherapyValidationChain
from langchain.memory.chat_message_histories import MongoDBChatMessageHistory
import os

def initialize_mental_healthcare_agent(session_id="123456"):
    # Initialize the chains
    conversation_utterance_chain = TherapyConversationChain.from_llm(
        global_dict['llm'], verbose=True
    )

    validation_chain = TherapyValidationChain.from_llm(
        global_dict['llm'], verbose=True
    )

    # Set up MongoDB for agent and actual conversation histories
    mongodb_for_agent = MongoDBChatMessageHistory(
        connection_string=os.getenv("MONGODB_CONNECTION_STRING"),
        session_id=f"therapy_agent_{session_id}",
        database_name="therapy-agent-chat-messages",
        collection_name="therapy-agent-history"
    )

    mongodb_for_actual = MongoDBChatMessageHistory(
        connection_string=os.getenv("MONGODB_CONNECTION_STRING"),
        session_id=f"therapy_actual_{session_id}",
        database_name="therapy-actual-chat-messages",
        collection_name="therapy-actual-history"
    )

    # Initialize conversational memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        output_key="output",
        chat_memory=mongodb_for_agent,
        k=3,
        return_messages=True
    )

    # Initialize the agent
    agent = initialize_agent(
        agent="chat-conversational-react-description",
        tools=tools_diff,
        llm=global_dict['llm'],
        verbose=True,
        max_iterations=2,
        early_stopping_method='generate',
        memory=conversational_memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        agent_kwargs={
            'prefix': PREFIX,
            'format_instructions': FORMAT_INSTRUCTIONS,
            'suffix': SUFFIX
        }
    )

    # Create a new prompt for the agent
    new_prompt = agent.agent.create_prompt(
        system_message=PREFIX,
        tools=tools_diff
    )
    
    # Store the agent and related information in a global unique dictionary
    global_unique[session_id] = {
        "conv_agent": agent,
        "actual_memory": mongodb_for_actual,
        "product_questions": 0,
        "session_scheduled": False,
        "greeted": False
    }
    
    # Update the agent's prompt
    agent.agent.llm_chain.prompt = new_prompt

    # Update global dictionaries with the new chains and memory
    global_dict["agent_memory"] = conversational_memory
    global_dict["validation_chain"] = validation_chain
    global_dict['conversation_utterance_chain'] = conversation_utterance_chain

# Make sure to call this function to initialize the agent
# initialize_mental_healthcare_agent()
