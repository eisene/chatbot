from duffel_api.models import Offer
from typing import List, Callable, Any


def _generic_ranker(offers: List[Offer], calc_rank: Callable[[Offer], Any]) -> List[Offer]:
    ranked = [(offer, calc_rank(offer)) for offer in offers]
    return [x[0] for x in sorted(ranked, key=lambda x: x[1])]


def price_ranker(offers: List[Offer]) -> List[Offer]:
    offers = filter(lambda o: o.total_currency == 'USD', offers)
    return _generic_ranker(offers, lambda offer: float(offer.total_amount))


def departure_time_ranker(offers: List[Offer]) -> List[Offer]:
    """Ranks a list of Duffel offers by their departure times, with an optional blackout time block."""
    return _generic_ranker(offers, lambda offer: offer.slices[0].segments[0].departing_at)
