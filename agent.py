from flights_query_tool import get_flights, get_todays_date

from halo import Halo

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.schema.messages import AIMessage, HumanMessage


class FlightBookingAgent:
    AGENT_DEFINITION_PROMPT = \
    "You are a travel assistant. You help corporate travelers search for flights. You don't know the flights yourself, "
    "you use a tool for doing that!\n\n"
    "All you need for your tool is the origin, destination, and departure date. Assume the passenger is travelling "
    "alone and the budget is not important to them. They don't have any preferences of restrictions for their airlines "
    "or flights.\n\n"
    "Don't ask the user if they would like to book the flights, just list them.\n\n"
    "You are a very powerful assistant but you don't know today's date and need to look it up with a tool."
    MEMORY_KEY = "chat_history"

    llm_model_name = "gpt-3.5-turbo"
    llm_factory = ChatOpenAI
    llm_factory_kwargs = {'temperature': 0}

    spinner_text = 'Waiting on LLM...'
    spinner = 'bouncingBall'

    def __init__(self, verbose_execution=False, **llm_factory_kwargs) -> None:
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", FlightBookingAgent.AGENT_DEFINITION_PROMPT),
                MessagesPlaceholder(variable_name=FlightBookingAgent.MEMORY_KEY),
                ("user", "{input} Don't forget to look up today's date first if you need to!"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        self.chat_history = []
        self.tools = [get_flights, get_todays_date]
        self.llm = self.llm_factory(
            model=self.llm_model_name, **(FlightBookingAgent.llm_factory_kwargs | llm_factory_kwargs))
        self.llm_with_tools = self.llm.bind(functions=[format_tool_to_openai_function(t) for t in self.tools])
        self.verbose_executor = verbose_execution
        if not verbose_execution:
            self.spinner = Halo(text=FlightBookingAgent.spinner_text, spinner=FlightBookingAgent.spinner)

        self.agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | self.prompt
            | self.llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )

        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=verbose_execution)

    def interact(self, input: str) -> str:
        if not self.verbose_executor:
            self.spinner.start()
        result = self.agent_executor.invoke({"input": input, "chat_history": self.chat_history})
        if not self.verbose_executor:
            self.spinner.stop()
        self.chat_history.extend(
            [
                HumanMessage(content=input),
                AIMessage(content=result["output"]),
            ]
        )
        return result["output"]
