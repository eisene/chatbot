import argparse

from agent import FlightBookingAgent


examples = [
    [
        "I need a flight from New York to Chicago tomorrow.",
        "Actually, I would like to change the date to Nov 20th.",
    ],
    [
        "I need a flight from Paris to Toronto tomorrow.",
        "Actually, I would like to change the date to Nov 20th.",
    ],
    [
        "I need a flight from Paris to Toronto tomorrow.",
        "Actually, I would like to leave in seven days.",
    ],
    [
        "I need a flight from Paris to Toronto tomorrow.",
        "Actually, I would like to go to Kitchener, ON.",
        "Can we change the departure date to seven days from now?"
    ],
    # THESE ARE TOO HARD FOR GPT-3.5 AND NEED SOME FUNDAMENTAL CHANGES TO THE APPROACH HERE:
    # [
    #     "I need a flight from Paris to Toronto tomorrow.",
    #     "Actually, I meant the Cox Field airport in Paris, Texas.",
    #     "Can we change the departure date to seven days from now?"
    # ],
    # THIS ONE IS COMPLETELY IMPOSSIBLE, IT NEVER GETS IT RIGHT IF IT DOESN'T HAVE AN IATA LOOKUP TOOL:
    # [
    #     "I need a flight from Paris to Toronto tomorrow.",
    #     "Actually, I meant Paris, Texas.",
    # ],
    #
    # The problem is that Paris, TEXAS is LLM hell - because it is an approximate retreiver, it _really_ wants to round
    #   the question to the nearest thing it understands - some version of Paris, France
]


def main(verbose=False, llama2=False, example_num=1):
    if llama2:
        raise NotImplementedError("Needs more work before it's usable")
        # print("USING LLAMA2, MAKE SURE OLLAMA IS INSTALLED AND LLAMA2 IS PULLED!")
        # from langchain.chat_models import ChatOllama
        # FlightBookingAgent.llm_factory = ChatOllama
        # FlightBookingAgent.llm_model_name = "llama2"
    agent = FlightBookingAgent(verbose_execution=verbose)


    def interact(input):
        print("USER:", input)
        print("AGENT:", agent.interact(input))
        print("===")


    for input in examples[example_num - 1]:
        interact(input)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run.py',
        description='Flight booking LLM agent example'
    )
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--llama2', action='store_true')
    parser.add_argument('-e', '--example_num', default=1, type=int)
    args = parser.parse_args()
    main(args.verbose, args.llama2, args.example_num)
