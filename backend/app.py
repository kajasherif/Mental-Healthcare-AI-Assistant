import os
import json
import atexit
import shutil
from flask import Flask, request, jsonify, session
from langchain.callbacks import get_openai_callback

from agent import initialize_mental_healthcare_agent
from werkzeug.utils import secure_filename
from utils import (chat_with_therapy_agent,
                   intialize_model_and_vector_store,
                   global_unique
                   )
from flask_session import Session
from dotenv import load_dotenv

load_dotenv()


app = Flask(__name__)

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set")

os.environ["OPENAI_API_KEY"] = openai_api_key


UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

EVAL_FOLDER = os.path.join(os.getcwd(), 'evaluation')

if not os.path.exists(EVAL_FOLDER):
    os.makedirs(EVAL_FOLDER)

app.config['EVAL_FOLDER'] = EVAL_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SESSION_FILE_DIR'] = "./flask_session/"

Session(app)


def clear_session_directory(session_folder_path):
    if os.path.exists(session_folder_path):
        shutil.rmtree(session_folder_path)


atexit.register(clear_session_directory, app.config['SESSION_FILE_DIR'])


@app.route('/chat_with_agent', methods=['POST'])
def get_answer():
    try:
        query = request.json.get('query')
        name = request.json.get('name')
        session_key = request.json.get('conv_id')

        if session_key not in session:
            session[session_key] = False

        if not session[session_key]:
            intialize_model_and_vector_store()
            # Renamed to reflect the mental healthcare context
            initialize_mental_healthcare_agent(session_key)
            session[session_key] = True

        # Renamed to reflect the mental healthcare context
        res = chat_with_therapy_agent(name, query, session_key)

        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
