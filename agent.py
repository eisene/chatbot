import string

from halo import Halo
from flights_query_tool import get_flights, get_todays_date

from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import OutputParserException
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate


max_iata_attempts = 3


class UserRequestState(BaseModel):
    origin: str = Field(
        description="The origin of the flight", default="unknown"
    )
    destination: str = Field(
        description="The destination of the flight", default="unknown"
    )
    departure_date: str = Field(
        description="A string describing the departure date", default="unknown"
    )


class IATACodes(BaseModel):
    origin: str = Field(
        description="The IATA code of the nearest airport to the origin of the flight", default=""
    )
    destination: str = Field(
        description="The IATA code of the nearest airport to the destination of the flight", default=""
    )

    @validator("origin", allow_reuse=True)
    @validator("destination", allow_reuse=True)
    def single_word_in_all_caps(cls, field):
        if not all([c in string.ascii_uppercase for c in field]):
            raise ValueError("Badly formatted IATA, all IATA codes should consist only of upper case English letters.")
        return field


class FlightBookingAgent:
    AGENT_DEFINITION_PROMPT = \
    "You are a travel assistant. You help corporate travelers search for flights. You don't know the flights yourself, "
    "you use a tool for doing that!\n\n"
    "Always show the user the IATA codes of the origin and destination airports!\n\n"
    USER_PROMPT = \
    "{input} Remember that you can look up today's date if you need to. Don't forget the previous "
    "information I gave you, such as the departure date!"
    MEMORY_KEY = "chat_history"
    CURRENT_USER_REQUEST_KEY = "current_user_request"

    llm_model_name = "gpt-3.5-turbo"
    llm_factory = ChatOpenAI
    llm_factory_kwargs = {'temperature': 0}

    spinner_text = 'Waiting on LLM and Duffel...'
    spinner = 'bouncingBall'

    def __init__(self, verbose_execution=False, **llm_factory_kwargs) -> None:
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", FlightBookingAgent.AGENT_DEFINITION_PROMPT),
                MessagesPlaceholder(variable_name=FlightBookingAgent.MEMORY_KEY),
                MessagesPlaceholder(variable_name=FlightBookingAgent.CURRENT_USER_REQUEST_KEY),
                ("user", FlightBookingAgent.USER_PROMPT),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        self.chat_history = []
        self.tools = [get_flights, get_todays_date]
        self.llm = self.llm_factory(
            model=self.llm_model_name, **(FlightBookingAgent.llm_factory_kwargs | llm_factory_kwargs))
        self.llm_with_tools = self.llm.bind(functions=[format_tool_to_openai_function(t) for t in self.tools])

        self.current_request = UserRequestState()

        self.verbose_executor = verbose_execution
        if not verbose_execution:
            self.spinner = Halo(text=FlightBookingAgent.spinner_text, spinner=FlightBookingAgent.spinner)

        request_parser = PydanticOutputParser(pydantic_object=UserRequestState)
        self.current_request_chain = (
            PromptTemplate(
                template="The user was looking for a flight from origin \"{origin}\" to destination \"{destination}\", "
                    " with departure date \"{departure_date}\". They provided new instructions: {query}. What origin, "
                    "destination, and departure date does the user want now?\n"
                    "{format_instructions}\n"
                    "Reply with only the JSON object.",
                input_variables=["query", "origin", "destination", "departure_date"],
                partial_variables={"format_instructions": request_parser.get_format_instructions()},
            )
            | self.llm
            | request_parser
        )

        iata_parser = PydanticOutputParser(pydantic_object=IATACodes)
        self.iata_conversion_chain = (
            PromptTemplate(
                template="What are the IATA codes of the airports closest to the following two locations: "
                    "origin=\"{origin}\", destination=\"{destination}\"?\n"
                    "{format_instructions}\n"
                    "Reply with only the JSON object.\n"
                    "{val_msg}",
                input_variables=["origin", "destination", "val_msg"],
                partial_variables={"format_instructions": iata_parser.get_format_instructions()},
            )
            | self.llm
            | iata_parser
        )

        self.search_agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
                FlightBookingAgent.MEMORY_KEY: lambda x: x[FlightBookingAgent.MEMORY_KEY],
                FlightBookingAgent.CURRENT_USER_REQUEST_KEY: lambda x: x[FlightBookingAgent.CURRENT_USER_REQUEST_KEY],
            }
            | self.prompt
            | self.llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )

        self.search_agent_executor = AgentExecutor(
            agent=self.search_agent, tools=self.tools, verbose=verbose_execution, handle_parsing_errors=True)

    def interact(self, input: str) -> str:
        if not self.verbose_executor:
            self.spinner.start()

        if self.verbose_executor:
            print("Parsing current user request state")
        self.current_request = self.current_request_chain.invoke(
            {
                "query": input,
                "origin": self.current_request.origin,
                "destination": self.current_request.destination,
                "departure_date": self.current_request.departure_date,
            }
        )

        if self.verbose_executor:
            print("Finding IATA codes")
        succeeded, val_msg = False, ""
        for i in range(max_iata_attempts):
            if self.verbose_executor:
                print("    ...attempt 1 out of " + str(max_iata_attempts))
            try:
                iata_codes = self.iata_conversion_chain.invoke(
                    {
                        "origin": self.current_request.origin,
                        "destination": self.current_request.destination,
                        "val_msg": val_msg,
                    }
                )
                succeeded = True
                break
            except OutputParserException as err:
                val_msg = f"It seems something was wrong in your response. The error message is \"{str(err)}\". " \
                          f"Could you try to fix your response?"
        if not succeeded:
            raise ValueError(f"Cannot produce valid IATA codes after {max_iata_attempts} tries")

        if self.verbose_executor:
            print("Kicking off flight finder agent")
        result = self.search_agent_executor.invoke(
            {
                "input": input,
                FlightBookingAgent.MEMORY_KEY: self.chat_history,
                FlightBookingAgent.CURRENT_USER_REQUEST_KEY: [
                    HumanMessage(
                        content="Here is what we know about the user's request: " + self.current_request.json()
                    ),
                    HumanMessage(
                        content="Here are the IATA codes for the origin and destination flights: " + iata_codes.json()
                    )
                ],
            }
        )
        if not self.verbose_executor:
            self.spinner.stop()
        self.chat_history.extend(
            [
                HumanMessage(content=input),
                AIMessage(content=result["output"]),
            ]
        )
        return result["output"]
