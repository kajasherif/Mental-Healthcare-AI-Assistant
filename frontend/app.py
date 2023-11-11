import streamlit as st
import threading
import datetime
import requests
import json
import time
import os
import uuid
import base64
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain import PromptTemplate, LLMChain

from streamlit.runtime.scriptrunner import get_script_run_ctx
ctx = get_script_run_ctx()
session_id = ctx.session_id
print("session_id:", session_id)


def main():
    st.title("Mental Healthcare AI Assistant")
    st.write("A compassionate companion for your mental wellness journey. Feel free to share your thoughts and questions.")

    # Initialize chat messages if not present in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "captured_response" not in st.session_state:
        st.session_state.captured_response = []

    with st.sidebar:
        if len(st.session_state.captured_response) > 0:
            st.sidebar.title("Therapy Session Request:")
            st.sidebar.json(st.session_state.captured_response)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input('Enter your question:')

    if question:
        start_time = time.time()
        print("--->>>", question)
        # Process the question (whether from text or audio)
        process_question(question)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("elapsed_time: ", elapsed_time)
        # st.write(f"Time taken for the response: {elapsed_time:.2f} seconds")


llm = AzureChatOpenAI(
    openai_api_base="https://deepan-oai.openai.azure.com/",
    openai_api_version="2023-07-01-preview",
    deployment_name="gpt-35-turbo",
    openai_api_key="5fac4eb8008d44fd9619995aaed3b085",
    openai_api_type="azure",
    temperature=0
)


def get_answer_from_llm(question):
    headers = {
        'Content-Type': 'application/json',
        'Cookie': 'session=c4c18a76-358e-4148-8c09-d10df71f0d93'
    }

    payload = json.dumps({
        "query": question,
        "name": "Sherief",
        "openai_api_base": "https://jason-oai.openai.azure.com/",
        "redis_endpoint": "jason-soligen.redis.cache.windows.net",
        "conv_id": f"aibo-bot-{session_id}"
    })

    response = requests.post(
        'http://localhost:5000/chat_with_agent', headers=headers, data=payload)

    print("Get Answer from Backend: ", response.text)
    return response


def Find_Emotion(question):
    instruct = " Given a user's question, determine if the emotion conveyed is either \"angry\" or \"frustration\". If neither emotion is detected, respond with \"neutral\". Provide a one-word response based on the emotion detected."
    temp = instruct + "sentence: {question} sentiment:"
    prompt_template1 = PromptTemplate(
        template=temp, input_variables=["question"])
    llm_chain1 = LLMChain(prompt=prompt_template1, llm=llm, verbose=False)
    answer = llm_chain1.run(question)
    return answer


def process_question(question):
    results = {}
    translated_formatted_response = ""
    # Emotion Detect
    # emotion_detected = Find_Emotion(question)

    # if emotion_detected in ["angry", "frustration"]:
    #     with st.chat_message("user"):
    #         st.markdown(question)
    #         st.session_state.messages.append(
    #             {"role": "user", "content": question})

    #     time.sleep(2)
    #     with st.chat_message("assistant"):
    #         # response_message = f"You are in {emotion_detected}, I'll connect to a live agent."
    #         response_message = f"I understand that you are frustrated and angry. I'm truly sorry for any inconvenience you've experienced. I will transfer you to a live agent."
    #         st.markdown(response_message)
    #         st.session_state.messages.append(
    #             {"role": "assistant", "content": response_message})
    #     return

    def fetch_answer():
        response = get_answer_from_llm(question)
        response_json = response.json()
        if 'output' in response_json:
            results['formatted_response'] = response_json['output']
            print("fetch answer() -> llm response: ",
                  results['formatted_response'])

        if "demo" in response_json:
            results["demo"] = response_json["demo"]
            print("="*10, results['demo'])

    with st.chat_message("user"):
        st.markdown(question)
        st.session_state.messages.append(
            {"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner('Fetching an answer...'):
            thread1 = threading.Thread(target=fetch_answer)
            thread1.start()
            thread1.join()

        # Check if we got a response
        if 'formatted_response' in results:
            if "demo" in results:
                print("=*"*10, results['demo'])
                st.session_state.captured_response.append(results['demo'])

            # st.session_state.live_agent_messages.append(
            #     {"role": "assistant", "content": results['formatted_response']})
            st.session_state.messages.append(
                {"role": "assistant", "content": results['formatted_response']})
            st.markdown(results['formatted_response'])

        else:
            st.warning("Unexpected response from the server. Please try again.")
