from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor

from dotenv import load_dotenv
import os

def configure():
    load_dotenv()
    return os.getenv('OPENAI_API_KEY')

model = ChatOpenAI(
    model = "gpt-4-turbo",
    api_key=configure(),
)

db = SQLDatabase.from_uri('sqlite:///ipca.db')

toolkit = SQLDatabaseToolkit(
    llm=model,
    db=db,
)

system_message = hub.pull('hwchase17/react')

agent = create_react_agent(
    llm=model,
    tools=toolkit.get_tools(),
    prompt=system_message,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True,
)

prompt = '''
    Use as ferrmentas necessárias para responder perguntas relacionadas ao histórico de IPCA ao longo dos anos.
    Responda tudo em português brasileiro.
    Perguntas: {input}
'''

prompt_template = PromptTemplate.from_template(prompt)

question = '''
    Baseado nos dados históricos de IPCA, faça uma previsão dos valores de 
    IPCA de cada mês até o final de 2024.
'''
output = agent_executor.invoke({
    'input': prompt_template.format(input=question)
})

print(output.get('output'))

