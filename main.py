import chromadb
from embedding import GeminiEmbeddingFunction
# from llm import model
import time
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
chromodb_NAME = os.environ['DB_NAME']

embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = False

chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection(name=chromodb_NAME, embedding_function=embed_fn)

GOOGLE_API = os.environ['GOOGLE_API_KEY']
genai.configure(api_key=GOOGLE_API)

model = genai.GenerativeModel("gemini-1.5-flash-8b-exp-0924")
def context_retrival(query):
    
    result = db.query(query_texts=[query], n_results=3)
    [passage]= result["documents"]
    passage = ' '.join(passage)
    passage = passage.replace("\n", " ")
    query = query.replace("\n", " ")
    return query, passage


def get_answer(query, context):
    
    prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below. 
                    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
                    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
                    strike a friendly and converstional tone. If the passage is irrelevant to the answer, you may ignore it.

                    QUESTION: {query}
                    CONTEXT: {context}
                    
                    """

    answer = model.generate_content(prompt)
    return answer.text

    


if __name__ == '__main__':

    question_list = [ 
                    "What are the reasons for ECB leaving rates unchanged?",
                    "What are the indicators to assume growth in the retail sector of the economy?",
                    "Can you compare the performance of Chinese economy with the US economy?",
                    "Who is IMF's chief economist and what did he say about Chinese economy?",
                    "What steps are taken to minimize the role of China?",
                    "What is the impact of weakness in the housing sector in the US?",
                    "Can you highlight the factors contributing to the increase in jobs in the US?",
                    "Please summarize the state of Germany's economy?",
                    "What is the state of Chinese exports?"]
    for ques in question_list:
        query , context = context_retrival(ques)
        answer = get_answer(query,context)
        
        print('QUERY:  ', query)
        print('ANSWER:  ', answer)
        print('-'*80)
        time.sleep(10)
