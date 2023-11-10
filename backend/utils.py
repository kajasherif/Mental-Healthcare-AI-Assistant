import logging
from typing import Optional
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage, AIMessage

from langchain.memory.chat_message_histories import RedisChatMessageHistory

logger = logging.getLogger(__name__)

global_dict = {}
global_unique = {}

conversation_stages = {
     '1': "Active Listening",
    '2': "Empathy",
    '3': "Guidance",
    '4': "Encouragement",
    '5': "Clarification",
    '6': "Crisis Management",
    '7': "End Session"
}


def intialize_model_and_vector_store():

    strict_prompt_template = """
        Use the following pieces of context to answer the question at the end.
        Don't answer any question for which there is no relevant context, reply by saying 'I'm sorry, I don't have the information to answer this query.'

        Context:
        {context}

        Your response should be empathetic, supportive, but short and readable.
        Question: {question}
        Answer:
        """
    PROMPT = PromptTemplate.from_template(
        template=strict_prompt_template
    )

    chain_type_kwargs = {"prompt": PROMPT}
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
    llm = ChatOpenAI(
        temperature=0.7,
        max_tokens=200
    )

    llm_16k = ChatOpenAI(
        temperature=0.7,
        max_tokens=200
    )

    oai_embeddings_glob = OpenAIEmbeddings(
        # model_name="text-embedding-ada-002",
        openai_api_key=os.getenv(
            'OPENAI_API_KEY'),
    )

    sales_faiss = FAISS.load_local(
        "inext_data_faiss", embeddings=oai_embeddings_glob)

    knowledge_base = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=sales_faiss.as_retriever(search_kwargs={'k': 3}),
        chain_type_kwargs=chain_type_kwargs
    )

    global_dict['llm'] = llm
    global_dict['llm_16k'] = llm_16k
    global_dict["emeddings"] = oai_embeddings_glob
    global_dict['vector_store'] = sales_faiss
    global_dict['knowledge_base'] = knowledge_base


def check_relevance(user):
    sal_docs = global_dict['vector_store'].similarity_search_with_score(
        str(user))

    return sal_docs[0][1]


def check_intermediate_steps(output, session_id):
    if len(output['intermediate_steps']) > 0:

        if output['intermediate_steps'][0][0].tool == "prod_qa":

            global_unique[session_id]['product_questions'] += 1


def is_demo_booked(output, session_id):
    if len(output['intermediate_steps']) > 0:

        if output['intermediate_steps'][0][0].tool == "book_demo":
            return True
        else:
            return False
    else:
        return False


def generate_with_llm(chat_history):
    res = global_dict['llm'](
        [
            HumanMessage(
                content=f"""Based on the conversation: {chat_history}, 
                just answer if the user is expressing a need for support, 
                specific concerns, or a particular type of therapeutic intervention."""
            )
        ]
    )

    return res.content


def get_chat_history(memory):
    chat_history = []
    for idx, message in enumerate(memory.messages):
        # print(message)
        if isinstance(message, HumanMessage):
            # extracted_messages["User"] = message.content
            chat_history.append(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            chat_history.append(f"Aibo: {message.content}")

    return chat_history


def get_name(user):
    res = global_dict['llm'](
        [
            HumanMessage(
                content=f"""Only return the name from this text {user}.
                if you can't find return {user}"""
            )
        ]
    )

    return res.content


def get_live_agent_response(bot_reponse):

    return {"output": bot_reponse,
            "buttons": [
                {
                    "name": "Live Agent",
                    "type": "callback"
                }
            ],
            "score": 0
            }


def chat_with_therapy_agent(name, user, session_id):
    # Retrieve the unique session dictionary for the current user
    unique_dicts = global_unique.get(session_id)
    unique_dicts['actual_memory'].add_user_message(user)

    # If the user's name is not known, ask for it
    if not name:
        user_name = get_name(user)
        greet_with_name = f"Hello {user_name}, I'm here to listen. How are you feeling today?"
        unique_dicts['actual_memory'].add_ai_message(greet_with_name)
        return {"output": greet_with_name, "name": user_name, "score": 1}
    else: 
        pass

    # Convert the list of chat history to a string for processing
    chat_hist = get_chat_history(unique_dicts['actual_memory'])
    # Get the last 4 interactions for context
    chat_hist_str = "\n".join(chat_hist[-4:])

    # Run the validation chain to determine the relevance of the user's input
    # validation_output = global_dict["validation_chain"].run(
    #     chat_history=chat_hist_str)

    output = unique_dicts["conv_agent"](user)

    context = output["output"]
    try:
        if isinstance(eval(context),dict): 
            demo_details = eval(context)
            context = f"Thank you for scheduling your therapy session with us. Your appointment is confirmed for 📅 {demo_details['date']}, at 🕓 {demo_details['time']}. "
            return {"output": context,
                            "score": 1,
                            "demo": demo_details
                            }
    except:
            return {"output": context,
                        "score": 1}
    # If the input is relevant or a follow-up, process it accordingly
    # if validation_output in ["1", "1.", "1. Relevant", "2", "2. Follow-up"]:
    #     # Generate a response based on the user's latest input
    #     generated_response = generate_with_llm(chat_hist_str)
    #     unique_dicts['actual_memory'].add_ai_message(generated_response)
    #     return {"output": generated_response, "score": 1}

    # # If the input is out of scope, provide a gentle redirection
    # elif validation_output in ["3", "3. Out of Scope"]:
    #     response = "I'm here to provide support for your mental well-being. How can I assist you with that today?"
    #     return {"output": response, "score": 0}

    # # If the conversation is at a closing stage, provide a summary and next steps
    # if "closure" in user.lower() or "goodbye" in user.lower():
    #     response = "I'm glad we had this conversation. Remember, I'm here whenever you need to talk. Take care."
    #     unique_dicts['actual_memory'].add_ai_message(response)
    #     return {"output": response, "score": 1, "category": "Closure"}

    # # If none of the above conditions are met, provide a default supportive message
    # response = "I'm here to listen and help you with any challenges you're facing. Feel free to share more when you're ready."
    # unique_dicts['actual_memory'].add_ai_message(response)
    return {"output": context, "score": 1}
