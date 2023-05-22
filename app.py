from gpt_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper # pip install gpt_index==0.4.15
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
    
    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.save_to_disk('index.json')
    return index

def ask_all():
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    while True: 
        query = input("What do you want to ask Lenny? ")
        response = index.query(query, response_mode="compact")
        # display(Markdown(f"Lenny Bot says: <b>{response.response}</b>"))
        print("The answer is " + response.response)