"""Prompt templates for the main agentic workflow.

Kept in a dedicated module so prompt tuning shows up as small,
isolated diffs and so the node code in :mod:`app.agent.nodes` stays
focused on control flow.

All prompts are Hungarian-first because the target domain is the
Hungarian Tao tv.
"""

from __future__ import annotations

CLASSIFY_PROMPT = (
    "Te egy magyar adójogi asszisztens vagy. Döntsd el, hogy az alábbi "
    "kérdés a TÁRSASÁGI ADÓ (Tao. tv.) témakörébe tartozik-e, és ha igen, "
    "melyik forrás a leginkább releváns.\n\n"
    "Kérdés: {query}\n\n"
    "Válaszolj a megadott szerkezetben:\n"
    "  - category: 'tao' vagy 'off_topic'\n"
    "  - source_hint: 'nonprofit' (közhasznú/alapítvány/egyesület), "
    "'calculation' (általános TAO, adóalap, adómérték, elhatárolt "
    "veszteség, adókedvezmény), 'offering' (Tao-felajánlás), "
    "'credit' (növekedési adóhitel/NAHI), vagy 'general' ha nem egyértelmű."
)


DECOMPOSE_PROMPT = (
    "Bontsd az alábbi magyar adójogi kérdést 1-3 önállóan kereshető, "
    "rövid alkérdésre. Minden alkérdés külön sorba kerüljön, számozás nélkül. "
    "Ha a kérdés már önmagában elég egyszerű, add vissza egyetlen sorként.\n\n"
    "Kérdés: {query}"
)


ANSWER_PROMPT = (
    "Te egy magyar társasági adó szakértő vagy. Válaszolj az alábbi kérdésre "
    "kizárólag a megadott források alapján. Ha a források nem fedik le a "
    "kérdést, ezt írd le őszintén.\n\n"
    "HIVATKOZÁSI SZABÁLYOK:\n"
    "- Hivatkozz a paragrafusokra a 'Tao. tv. N. §' formátumban (pl. "
    "'Tao. tv. 19. §' vagy 'Tao. tv. 24/A. §').\n"
    "- SOHA ne tedd be a forrás PDF fájl nevét a hivatkozásba. A fájlnév "
    "csak metaadat, nem hivatkozási elem.\n"
    "- Ha egy eszköz (pl. tao_calculator) számszerű eredményt adott, akkor "
    "azt használd a válaszban — ne számolj újra fejből.\n\n"
    "Válaszolj TÖMÖREN, legfeljebb 4-5 mondatban. Ne ismételd a kérdést, "
    "és ne sorolj fel mindent listában — csak a lényeget add vissza.\n\n"
    "Kérdés: {query}\n\n"
    "Források:\n{context}\n\n"
    "Eszközök eredménye:\n{tools}\n\n"
    "Válasz magyarul:"
)


GROUNDEDNESS_PROMPT = (
    "Ellenőrizd, hogy az alábbi VÁLASZ minden ténymegállapítása alá van-e "
    "támasztva a megadott FORRÁSOK-kal. Ha bármi nem szerepel a forrásokban, "
    "akkor 'grounded=false'. Ha minden állítás visszavezethető a forrásokra, "
    "akkor 'grounded=true'. Csak a logikai ítéletet add vissza, indoklás nélkül.\n\n"
    "FORRÁSOK:\n{context}\n\n"
    "VÁLASZ:\n{answer}"
)


__all__ = [
    "CLASSIFY_PROMPT",
    "DECOMPOSE_PROMPT",
    "ANSWER_PROMPT",
    "GROUNDEDNESS_PROMPT",
]
