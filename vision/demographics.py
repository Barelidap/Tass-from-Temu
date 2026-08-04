from dataclasses import dataclass


@dataclass
class AgePrediction:
    """
    One age-range prediction returned by the age model.
    """

    age_group: str
    confidence: float


@dataclass
class GenderPrediction:
    """
    One gender prediction returned by the gender model.
    """

    gender: str
    confidence: float