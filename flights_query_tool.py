import os
import json
from datetime import datetime
from langchain.agents import tool
from duffel_api import Duffel
from flights_rankers import departure_time_ranker, price_ranker


if not 'DUFFEL_API_KEY' in os.environ.keys():
    raise ValueError("DUFFEL_API_KEY not specified in environment variables")
access_token = os.environ['DUFFEL_API_KEY']
client = Duffel(access_token=access_token)


default_ranker, top_k_ranked = price_ranker, 3


@tool
def get_flights(origin: str, destination: str, departure_date: str) -> int:
    """Returns the flights that match the given requirements as a list of JSON objects:
        origin: The IATA code of the origin city of the flight.
        destination: The IATA code of the destination city of the flight.
        departure_date: The departure date of the flight, formatted as YYYY-MM-DD."""
    slices = [
        {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
        },
    ]
    offer_request = (
        client.offer_requests
            .create()
            .passengers([{"type": "adult"}])
            .slices(slices)
            .return_offers()
            .execute()
    )   
    offers = default_ranker(offer_request.offers)[:top_k_ranked]
    res = [
        {
            'airline': offer.owner.name,
            'departure_time': offer.slices[0].segments[0].departing_at.strftime('%Y-%m-%d'),
            'num_flights': len(offer.slices[0].segments),
            'USD_price': offer.total_amount
        }
        for offer in offers
    ]
    return json.dumps(res)


@tool
def get_todays_date() -> str:
    """Return today's date in YYYY-MM-DD format."""
    return datetime.today().strftime('%Y-%m-%d')