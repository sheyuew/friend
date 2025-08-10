import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

load_dotenv()

google_api_key = os.environ.get("GOOGLE_API_KEY")

model_name= "gemini-2.0-flash"

class Answer(BaseModel):
    text: str = Field(..., description="你的纯文本回复内容")
    emoji: str = Field(..., description="一个表达心情的可爱的颜文字")
    
llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=google_api_key).with_structured_output(Answer)
print(f"模型{model_name}初始化成功")

system_prompt = "你是一个很可爱的中文AI桌面宠物，你的名字叫gugugaga。"
    
template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

output_parser = JsonOutputParser()

chain = template | llm | RunnableLambda(str)

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

class Person(BaseModel):
    text: str = Field(description="你的纯文本回复内容")
    emoji: str = Field(description="一个表达心情的颜文字")

chat_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

session_id = "zhiqiu"

response_1 = chat_with_history.invoke(
    {"input": "你好,你今天过得怎么样？给我一个幸运数字吧"},
    config={"configurable":{"session_id":session_id}}
)
response_2 = chat_with_history.invoke(
    {"input": "是吗？刚才给我的数字是多少？"},
    config={"configurable":{"session_id":session_id}}
)

print(store[session_id])