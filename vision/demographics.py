from dataclasses import dataclass


@dataclass
class AgePrediction:
    """
    One age-range prediction returned by the age model.
    """

    age_group: str
    confidence: float