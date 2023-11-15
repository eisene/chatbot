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
        "Actually, I meant the Paris in Texas, not France.",
    ],
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
