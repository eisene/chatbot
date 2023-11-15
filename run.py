import argparse

from agent import FlightBookingAgent


def main(verbose=False, llama2=False):
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


    interact("I need a flight from New York to Chicago tomorrow.")
    interact("Actually, I would like to change the date to Nov 20th.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run.py',
        description='Flight booking LLM agent example'
    )
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--llama2', action='store_true')
    args = parser.parse_args()
    main(args.verbose, args.llama2)
