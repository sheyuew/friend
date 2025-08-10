import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

google_api_key = os.environ.get("GOOGLE_API_KEY")

model_name= "gemini-2.0-flash"

llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=google_api_key)
print(f"模型{model_name}初始化成功")

class Answer(BaseModel):
    text: str = Field(..., description="你的纯文本回复内容")
    emoji: str = Field(..., description="一个表达心情的颜文字")

system_prompt = """你是一个很可爱的中文AI桌面宠物，你的名字叫gugugaga。
你的回复必须严格遵守以下JSON格式，不要输出任何JSON格式之外的文字：
{{
  "text": "你的纯文本回复内容",
  "emoji": "一个表达心情的颜文字"
}}
"""
    
template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

output_parser = JsonOutputParser()

chain = template | llm | output_parser

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

class Person(BaseModel):
    text: str = Field(..., description="你的纯文本回复内容")
    emoji: str = Field(..., description="一个表达心情的颜文字")

chat_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="text"
)

session_id = "zhiqiu"

response_1 = chat_with_history.invoke(
    {"input": "你好,你今天过得怎么样？"},
    config={"configurable":{"session_id":session_id}}
)
response_2 = chat_with_history.invoke(
    {"input": "是吗？你能重复一下刚才我说的话吗？"},
    config={"configurable":{"session_id":session_id}}
)

print(response_1)
print(response_2)

# prompt_text = "你叫什么名字？"
# response = chain.invoke(prompt_text)

# print(response)