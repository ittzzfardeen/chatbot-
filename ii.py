from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os 


load_dotenv()
api_key = os.getenv("groq_api_key")
os.environ["GROQ_API_KEY"] = api_key


print("befor model loading")
model="llama-3.1-8b-instant"
groq_model=ChatGroq(model=model)
kk=groq_model.invoke("what is the capital of france?")
print(kk)