"""Query mix used in the load test.

Deliberately mixes question types to give a realistic distribution of
agent paths: short factual lookup, calculation, off-topic refusal,
multi-section retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass

_BASE_QUERIES: list[tuple[str, str]] = [
    ("rate", "Mennyi a társasági adó mértéke?"),
    ("loss", "Hány százalékig vehető figyelembe az elhatárolt veszteség?"),
    ("loss_years", "Hány évre vihető előre az elhatárolt veszteség?"),
    ("calc_small", "Mennyi a TAO 500 000 Ft adóalap után?"),
    ("calc_medium", "Mennyi a TAO 5 000 000 Ft adóalap után?"),
    ("calc_with_loss", "Mennyi a TAO 10 000 000 Ft adóalap és 3 000 000 Ft elhatárolt veszteség mellett?"),
    ("citation_check", "Mit mond a 19. § a társasági adó mértékéről?"),
    ("nahi", "Mi a növekedési adóhitel lényege?"),
    ("offering", "Mekkora lehet a TAO-felajánlás maximális összege?"),
    ("nonprofit", "Hogyan adózik egy közhasznú nonprofit gazdasági társaság?"),
    ("off_topic_weather", "Milyen az időjárás Budapesten?"),
    ("off_topic_recipe", "Hogyan készítsek brassói aprópecsenyét?"),
]


@dataclass
class ScenarioQuery:
    label: str
    query: str
    expected_category: str  # "tao" / "off_topic"


def build_query_mix(n: int) -> list[ScenarioQuery]:
    """Return `n` queries cycling through the base mix."""
    out: list[ScenarioQuery] = []
    for i in range(n):
        label, query = _BASE_QUERIES[i % len(_BASE_QUERIES)]
        category = "off_topic" if label.startswith("off_topic") else "tao"
        out.append(ScenarioQuery(label=label, query=query, expected_category=category))
    return out


__all__ = ["ScenarioQuery", "build_query_mix"]
