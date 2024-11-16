import logging
from app import create_app
from app.llm import init_llm
from app.index import create_index, load_index
from app.routes import blueprint, initialize_app
from app.config import HTTP_PORT, INDEX_PERSIST_DIRECTORY, INIT_INDEX

logging.basicConfig(level=logging.INFO)

app = create_app()
app.register_blueprint(blueprint)

if __name__ == "__main__":
    embed_model = init_llm()
    
    if INIT_INDEX:
        logging.info("Initializing index...")
        create_index(embed_model, INDEX_PERSIST_DIRECTORY)
    
    logging.info("Loading index...")
    initialize_app(embed_model, INDEX_PERSIST_DIRECTORY)
    
    app.run(host='0.0.0.0', port=HTTP_PORT, debug=True)
