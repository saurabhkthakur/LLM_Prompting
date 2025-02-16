import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API = os.environ['GOOGLE_API_KEY']

genai.configure(api_key=GOOGLE_API)

model = genai.GenerativeModel(
    "gemini-1.5-flash-8b-exp-0924",
    generation_config=genai.GenerationConfig(
        temperature=0.1,
        top_p=1,
    )
)
prompt = '''You are a helpful and informative bot that answers questions using text from the reference passage included below. 
                    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
                    However, you are talking to a non-technical audience, keep the answer short and highlight important points
                    . If the passage is irrelevant to the answer, you may ignore it.

                    QUESTION: Can you provide an overview of the US economy?
                    CONTEXT:  US economy shows strength. But will it last? The US economy experienced rapid growth in the third quarter. This was expected given 
the favorable monthly and higher frequency data that had been released in the past three months. Real (inflation-adjusted) GDP grew at an annual rate of 4.9%, the fastest growth since the fourth quarter of 2021. Over 80% of the increase in real GDP can be attributed to consumer spending and inventory accumulation. Business investment failed to grow, with rising investment in equipment and intellectual property offset by declining investment in equipment. Exports saw strong growth, as did government purchases. While extremely favorable, and not unexpected, many'''

print(model.count_tokens(prompt))
answer = model.generate_content(prompt , request_options={"timeout": 120} )
print(answer.text)






