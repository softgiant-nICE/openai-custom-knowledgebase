
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext, load_index_from_storage, StorageContext
from langchain import OpenAI
import sys
import os
# from IPython.display import Markdown, display

def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 300
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chenk size limit
    chunk_size_limit = 600

    #define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context
    )

    # index.save_to_disk('index.json') // for gpt-index
    index.storage_context.persist(persist_dir="persist_dir")
    return index

def ask_all():
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="persist_dir")
    # load index
    # index = GPTVectorStoreIndex.load_from_disk('index.json') //for gpt-index
    index = load_index_from_storage(storage_context)
    while True: 
        query = input("What do you want to ask Lenny? ")
        # response = index.query(query, response_mode="compact") //for gpt-index
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        # display(Markdown(f"Lenny Bot says: <b>{response.response}</b>"))
        print("The answer is " + response.response)


os.environ['OPENAI_API_KEY'] =  input('Paste your OpenAI key here and hit enter\n')

construct_index('./context_data/data')
ask_all()