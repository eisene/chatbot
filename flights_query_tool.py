import os
import json
from datetime import datetime
from langchain.agents import tool
from duffel_api import Duffel, ApiError
from flights_rankers import departure_time_ranker, price_ranker


if not 'DUFFEL_API_KEY' in os.environ.keys():
    raise ValueError("DUFFEL_API_KEY not specified in environment variables")
access_token = os.environ['DUFFEL_API_KEY']
client = Duffel(access_token=access_token)


default_ranker, top_k_ranked = price_ranker, 3

num_api_errors, max_api_errors = 0, 5
previous_lookups = set()


class TooManyApiErrorsException(Exception):
    pass


@tool
def get_flights(origin: str, destination: str, departure_date: str) -> int:
    """Returns the flights that match the given requirements as a list of JSON objects:
        origin: The IATA code of the origin city of the flight.
        destination: The IATA code of the destination city of the flight.
        departure_date: The departure date of the flight, formatted as YYYY-MM-DD.
    
    When composing your response based on the results of this tool, first write a sentece similar to these examples:
        \"Here are 3 flights from New York (JFK) to Toronto (YYZ) leaving on 2020-01-01\", or
        \"Here is 1 flight from Los Angeles (LAX) to Paris (CDG) leaving on 2022-10-15\"
    Then write down the flights using the following format for each flight:
        {index}.{airline} - Departs at {departure_time}, Number of connections: {num_connections}, Price: ${USD_price}
    """
    global num_api_errors, previous_lookups

    if (origin, destination, departure_date) in previous_lookups:
        return f"You already looked this up!"
    previous_lookups.add((origin, destination, departure_date))

    slices = [
        {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
        },
    ]
    try:
        offer_request = (
            client.offer_requests
                .create()
                .passengers([{"type": "adult"}])
                .slices(slices)
                .return_offers()
                .execute()
        )   
    except ApiError as err:
        num_api_errors += 1
        if num_api_errors >= max_api_errors:
            raise TooManyApiErrorsException
        return f"It seems something was wrong in your request. The error message is \"{err.message}\". Could you try " \
               f"to fix your request and then try using this tool again?"
    offers = default_ranker(offer_request.offers)[:top_k_ranked]
    res = [
        {
            'airline': offer.owner.name,
            'departure_time': offer.slices[0].segments[0].departing_at.strftime('%I:%M %p'),
            'num_connections': len(offer.slices[0].segments) - 1,
            'USD_price': offer.total_amount
        }
        for offer in offers
    ]
    return json.dumps(res)


@tool
def get_todays_date() -> str:
    """Return today's date in YYYY-MM-DD format."""
    return datetime.today().strftime('%Y-%m-%d')