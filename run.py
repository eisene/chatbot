from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from flights_query_tool import get_flights, get_todays_date
from langchain.schema.messages import AIMessage, HumanMessage


AGENT_DEFINITION_PROMPT = \
"You are a travel assistant. You help corporate travelers search for flights. You don't know the flights yourself, "
"you use a tool for doing that!\n\n"
"All you need for your tool is the origin, destination, and departure date. Assume the passenger is travelling alone "
"and the budget is not important to them. They don't have any preferences of restrictions for their airlines or "
"flights.\n\n"
"Don't ask the user if they would like to book the flights, just list them.\n\n"
"You are a very powerful assistant but you don't know today's date and need to look it up with a tool."
MEMORY_KEY = "chat_history"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", AGENT_DEFINITION_PROMPT),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input} Don't forget to look up today's date first if you need to!"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

chat_history = []
tools = [get_flights, get_todays_date]
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# llm = ChatOllama(model="llama2", temperature=0)
llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

input_text = "I need a flight from New York to Chicago tomorrow."
result = agent_executor.invoke({"input": input_text, "chat_history": chat_history})
chat_history.extend(
    [
        HumanMessage(content=input_text),
        AIMessage(content=result["output"]),
    ]
)

input_text = "Actually, I would like to change the date to Nov 20th."
result = agent_executor.invoke({"input": input_text, "chat_history": chat_history})
chat_history.extend(
    [
        HumanMessage(content=input_text),
        AIMessage(content=result["output"]),
    ]
)