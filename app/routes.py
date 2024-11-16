from flask import Blueprint, request, jsonify
import logging
from app.llm import init_llm
from app.index import load_index

blueprint = Blueprint("routes", __name__)
query_engine = None

def init_query_engine(index):
    global query_engine
    from llama_index.core import PromptTemplate
    
    template = (
        "Imagine you are an advanced AI expert in viticulture and plant health, "
        "with access to all current and relevant scientific research, agricultural guidelines, and expert recommendations.\n\n"
        "Following is the query and some additional parameters"
        "Query: {query_str}\n\n"
        "Here is some context related to the situation:\n"
        "-----------------------------------------\n"
        "{context_str}\n"
        "-----------------------------------------\n"
        "Based on the above information, please provide:\n"
        "1. A summary of the disease, its causes, and typical impact on grapevines.\n"
        "2. Recommended treatments and best practices to manage the disease under these environmental conditions. Give medicines to use specifically and how to use them\n"
        "3. Information taking into account the weather"
        "3. Preventative measures for avoiding such issues in the future.\n"
    )
    # template = (
    #     "{context_str}\n"

    #     "{query_str}"
    # )

    prompt_template = PromptTemplate(template=template)

    query_engine = index.as_query_engine(
        text_qa_template=prompt_template, 
        similarity_top_k=4
    )

@blueprint.route('/api/question', methods=['POST'])
def post_question():
    global query_engine
    
    json = request.get_json(silent=True)
    question = json['question']
    logging.info("Received question: `%s`", question)
    
    response = query_engine.query(question)
    return jsonify({'answer': response.response}), 200

def initialize_app(embed_model, persist_directory):
    index = load_index(embed_model, persist_directory)
    init_query_engine(index)
