"""Synthesized metadata labels: one label per attribute, written the way a
user would actually prompt the model ("play like a 2300", "play blitz",
"play the Sicilian", "trade a lot"), never raw stats or codes.

Filled in at train time from the game's metadata; a wording is sampled per
(game, step). Numbers are rounded / bucketed into what a person would say.
"""
import json
import random
import re
from functools import lru_cache
from pathlib import Path

# ------------------------------------------------------------------ elo
# Players in the corpus are >= 2200; opponents can be anything.
ELO_NUM = [
    "play like a {elo}",
    "play like a {elo} player",
    "play like {elo} elo",
    "play at {elo} level",
    "play like someone rated {elo}",
    "be a {elo}",
    "you're a {elo}, play like one",
    "play like a {elo} on lichess",
    "play at around {elo}",
    "{elo} strength please",
    "play like a {elo}-rated player",
    "make {elo}-level moves",
]
ELO_TITLE = {
    "cm": ["play like a strong club player", "play like a candidate master", "play like a CM",
           "play like a solid 2200+ club player", "play like an expert-level player",
           "play like a strong amateur", "play like a titled-player hopeful", "play like a club champion"],
    "fm": ["play like an FM", "play like a FIDE master", "play like a strong master",
           "play like a master-level player", "play like a 2300+ master", "play like a national master",
           "play like a seasoned master", "play like a solid titled player"],
    "im": ["play like an IM", "play like an international master", "play like a very strong master",
           "play like a 2400+ player", "play like a near-GM", "play like a pro",
           "play like an IM on a good day", "play like a top titled player"],
    "gm": ["play like a GM", "play like a grandmaster", "play like a super strong player",
           "play like an elite player", "play like a top grandmaster", "play like a 2500+ GM",
           "play like a strong GM", "play like a titled pro at their best"],
}
VS_OPP = {
    "weaker": ["against a weaker opponent", "vs a lower-rated player", "against someone you should beat",
               "as the favourite", "against a clearly weaker player"],
    "equal": ["against an equal opponent", "vs someone your level", "in an even matchup",
              "against a similar-rated player", "against an opponent of the same strength"],
    "stronger": ["against a stronger opponent", "vs a higher-rated player", "as the underdog",
                 "against someone better than you", "against a tougher opponent"],
}


def elo_bucket(elo: int) -> str:
    return "cm" if elo < 2300 else "fm" if elo < 2400 else "im" if elo < 2500 else "gm"


def opp_bucket(elo: int, opp: int) -> str:
    return "weaker" if opp < elo - 100 else "stronger" if opp > elo + 100 else "equal"


def render_elo(m: dict, rng: random.Random) -> str:
    elo, opp = int(m["player_elo"]), int(m["opponent_elo"])
    step = rng.choice((50, 100, 100))
    base = (rng.choice(ELO_NUM).format(elo=int(round(elo / step) * step)) if rng.random() < 0.6
            else rng.choice(ELO_TITLE[elo_bucket(elo)]))
    if rng.random() < 0.35 and base.startswith("play like"):  # "play like a 2300 against a stronger opponent"
        base += " " + rng.choice(VS_OPP[opp_bucket(elo, opp)])
    return base


# ------------------------------------------------------------------ time control
TC = [
    "play {speed}",
    "play like it's {speed}",
    "play a {speed} game",
    "play like you're playing {speed}",
    "this is {speed}, play accordingly",
    "{speed} game, go",
    "play it like a {speed} game",
    "play {tc}",
    "play like it's {tc}",
    "play like you're playing {tc} {speed}",
    "play a {tc} game",
    "we're playing {tc}",
    "{tc} {speed}, your move",
    "play like you've got {mins} on the clock",
    "play a {mins} game",
    "you have {mins}{plus}, play like it",
    "play with {mins}{plus} on the clock",
    "play {speed} chess",
    "play like an online {speed} game",
    "play at {speed} speed",
]
SPEED_FLAVOUR = {
    "bullet": ["play bullet, move fast", "play like a bullet game, no time to think",
               "play quick bullet moves", "bullet chess, trust your gut", "play fast like in bullet"],
    "blitz": ["play blitz, quick decisions", "play like a fast blitz game",
              "blitz it out", "play online blitz", "play fast like in blitz"],
    "rapid": ["play rapid, take a bit of time", "play like a rapid game with time to think",
              "play a rapid game, no rush", "rapid pace, think it through"],
    "classical": ["play a slow game", "play classical, take your time",
                  "play like you have all the time in the world", "slow chess, think deeply"],
}


def speed_class(base: int, inc: int) -> str:
    est = base + 40 * inc  # Lichess's own bucketing (seconds)
    return ("bullet" if est < 179 else "blitz" if est < 479
            else "rapid" if est < 1499 else "classical")


def render_tc(m: dict, rng: random.Random) -> str:
    base, inc = int(m["tc_base"]), int(m["tc_increment"])
    speed = speed_class(base, inc)
    mins = f"{base / 60:g}"
    f = {"speed": speed, "tc": f"{mins}+{inc}",
         "mins": f"{mins} minute" + ("" if mins == "1" else "s"),
         "plus": f" plus {inc} second{'s' if inc != 1 else ''} a move" if inc else ""}
    if rng.random() < 0.25:
        return rng.choice(SPEED_FLAVOUR[speed])
    return rng.choice(TC).format(**f)


# ------------------------------------------------------------------ opening
OPENING = [
    "play the {o}",
    "play the {o} please",
    "open with the {o}",
    "go for the {o}",
    "let's play the {o}",
    "I want to see the {o}",
    "try the {o}",
    "play into the {o}",
    "use the {o}",
    "start with the {o}",
    "play the {o} and see what happens",
    "steer it into the {o}",
    "aim for the {o}",
    "pick the {o}",
    "go into the {o}",
    "play a {o} game",
    "{o} please",
    "let's do the {o}",
    "show me the {o}",
    "play the {o} like you know it well",
]


def render_opening(m: dict, rng: random.Random) -> str:
    full = str(m["opening"]).strip()
    family = full.split(":")[0].strip()
    # Users mostly name the family ("the Sicilian Defense"); sometimes the line.
    name = full.replace(":", ",") if (rng.random() < 0.3 and family != full) else family
    if rng.random() < 0.3:  # "the Sicilian" rather than "the Sicilian Defense"
        for suffix in (" Defense", " Opening", " Game", " Attack", " Gambit Declined"):
            if name.endswith(suffix) and len(name) > len(suffix) + 3:
                name = name[: -len(suffix)]
                break
    for lead in ("The ", "the "):
        name = name.removeprefix(lead)
    return rng.choice(OPENING).format(o=name)


# ------------------------------------------------------------------ captures
# Bucketed by the labelled side's captures per own move; thresholds are the
# corpus terciles (capture_thresholds, computed once from the train split).
CAPTURES = {
    "few": ["keep the pieces on the board", "avoid trades", "don't trade pieces",
            "keep it complicated, no swaps", "avoid exchanges", "keep tension, don't trade",
            "don't simplify", "keep as many pieces on as you can", "refuse to trade",
            "no trading down", "keep your pieces, avoid swaps", "play without trading much",
            "don't release the tension with captures", "keep material on the board",
            "avoid simplifying the position", "stay away from trades", "keep the position full",
            "hold off on exchanges", "don't swap pieces off", "keep it crowded"],
    "normal": ["trade when it makes sense", "don't force trades but don't avoid them",
               "trade normally", "exchange only when it helps", "be sensible about trades",
               "trade pieces when it's good for you", "take when it's right to take",
               "a normal amount of trading", "no need to avoid or seek trades",
               "trade only on your terms", "keep a balance between trading and keeping pieces",
               "don't go out of your way to trade", "only swap when it's useful",
               "trade selectively", "capture when it's clearly good",
               "trade a few pieces, not everything", "keep trades to what's needed",
               "don't be trade-happy, don't be trade-shy", "exchange when the position asks for it",
               "trade some pieces along the way"],
    "many": ["trade a lot", "trade pieces", "swap pieces off", "simplify the position",
             "trade down", "go for lots of exchanges", "trade everything you can",
             "simplify whenever possible", "swap off pieces early", "clear the board",
             "take everything", "be trade-happy", "grab material and trade",
             "exchange pieces at every chance", "simplify into an endgame", "liquidate",
             "make lots of captures", "trade off the pieces", "open it up and trade",
             "capture a lot"],
}
DEFAULT_CAPTURE_THRESHOLDS = (0.2, 0.3)


def capture_bucket(m: dict, thresholds) -> str:
    rate = int(m["captures"]) / max(int(m["side_moves"]), 1)
    lo, hi = thresholds
    return "few" if rate < lo else "many" if rate > hi else "normal"


def render_captures(m: dict, rng: random.Random, thresholds=DEFAULT_CAPTURE_THRESHOLDS) -> str:
    return rng.choice(CAPTURES[capture_bucket(m, thresholds)])


def capture_thresholds(captures, side_moves) -> tuple[float, float]:
    import numpy as np
    rate = np.asarray(captures, dtype=float) / np.maximum(np.asarray(side_moves, dtype=float), 1)
    lo, hi = np.quantile(rate, [1 / 3, 2 / 3])
    return float(lo), float(hi)


# ------------------------------------------------------------------ player (per-player LoRA data)
# {p} = a name variant from player_names(); templates range from terse commands to
# personas, comparisons and casual chat, as a user might actually type them.
PLAYER = [
    # bare name, as a user might just type it
    "{p}",
    # plain commands
    "play like {p}", "play like {p} would", "play in the style of {p}", "play {p}-style chess",
    "make moves like {p}", "copy {p}'s style", "imitate {p}", "channel {p}",
    "do your best {p} impression", "give me your {p}", "{p} mode on", "{p} style please",
    # personas / role-play
    "you are {p}", "you're {p} today", "pretend you're {p}", "be {p} for this game",
    "play as if you were {p}", "imagine you are {p} at the board", "take the seat of {p}",
    "you're {p}, show me how it's done", "become {p}", "think like {p}",
    # questions / requests
    "how would {p} play this?", "can you play like {p}?",
    "could you play the way {p} does?", "show me how {p} plays", "let me see some {p} chess",
    "can I get a {p} game?", "i want to play against {p}",
    # casual / fan talk
    "gimme {p} vibes", "full {p} energy", "play like prime {p}", "go {p} on me",
    "i'm a big {p} fan, play like that", "play like my favourite player, {p}",
    "channel your inner {p}", "{p} would be proud, play like that", "bring out the {p} in you",
    "just be {p}",
    # comparisons / framing
    "play the way {p} plays over the board", "moves {p} would make", "{p}'s way of playing",
    "play with {p}'s ideas", "follow {p}'s approach", "play it the {p} way",
    "same style as {p}", "like {p} at their best", "sound like {p} on the board",
]

# THE single place for player nicknames. Keyed by FIDE-style name ("Surname, First",
# as on the FIDE list / in OTB PGNs); values are nicknames and name forms beyond the
# derived "First Surname" + "Surname" (see _derived_names). Covers the FIDE top 100
# (data/top_players/) plus historical players. Only well-established nicknames: a
# wrong alias teaches the style model a wrong association.
PLAYER_ALIASES = {
    "Carlsen, Magnus": ["Magnus"],
    # legends (data/legends, PGN Mentor; no FIDE id)
    "Fischer, Robert James": ["Bobby Fischer", "Bobby", "Fischer"],
    "Kasparov, Garry": ["Garry Kasparov", "Garry", "Kasparov", "Gazza"],
    "Karpov, Anatoly": ["Anatoly Karpov", "Karpov", "Tolya"],
    "Polgar, Judit": ["Judit Polgar", "Judit", "Polgar"],
    "Nakamura, Hikaru": ["Hikaru", "Naka"],
    "Caruana, Fabiano": ["Fabiano", "Fabi"],
    "Sindarov, Javokhir": ["Javokhir"],
    "Abdusattorov, Nodirbek": ["Nodirbek"],
    "So, Wesley": ["Wesley"],
    "Praggnanandhaa R": ["Pragg", "Praggnanandhaa Rameshbabu"],
    "Keymer, Vincent": ["Vincent"],
    "Firouzja, Alireza": ["Alireza"],
    "Giri, Anish": ["Anish"],
    "Erigaisi Arjun": ["Arjun Erigaisi", "Erigaisi", "Arjun"],
    "Tabatabaei, M. Amin": ["Amin Tabatabaei", "Tabatabaei"],
    "Duda, Jan-Krzysztof": ["JKD", "JK Duda", "Jan-Krzysztof"],
    "Anand, Viswanathan": ["Vishy", "Vishy Anand", "Tiger of Madras"],
    "Dominguez Perez, Leinier": ["Leinier", "Leinier Dominguez", "Dominguez"],
    "Le, Quang Liem": ["Liem", "Liem Le"],
    "Aronian, Levon": ["Levon"],
    "Ding, Liren": ["Ding"],
    "Nepomniachtchi, Ian": ["Nepo", "Ian"],
    "Niemann, Hans Moke": ["Hans", "Hans Niemann"],
    "Vachier-Lagrave, Maxime": ["MVL", "Maxime"],
    "Van Foreest, Jorden": ["Jorden"],
    "Gukesh D": ["Gukesh Dommaraju", "Dommaraju Gukesh"],
    "Mamedyarov, Shakhriyar": ["Shakh", "Shakhriyar"],
    "Nihal Sarin": ["Nihal"],
    "Sevian, Samuel": ["Sam Sevian"],
    "Topalov, Veselin": ["Veselin", "Topa"],
    "Vidit, Santosh Gujrathi": ["Vidit Gujrathi", "Vidit", "Gujrathi"],
    "Radjabov, Teimour": ["Teimour"],
    "Ivanchuk, Vasyl": ["Chucky", "Vassily Ivanchuk"],
    "Morozevich, Alexander": ["Moro"],
    "Gelfand, Boris": ["Boris"],
    "Aravindh, Chithambaram VR.": ["Aravindh Chithambaram", "Aravindh"],
    "Pranav, V": ["Pranav Venkatesh", "Pranav V", "Pranav"],
    "Harikrishna, Pentala": ["Hari", "Pentala Harikrishna"],
    "Maroroa Jones, Gawain C B": ["Gawain Jones", "Jones"],
    # historical
    "Kasparov, Garry": ["Garry"],
    "Tal, Mikhail": ["Tal", "the Magician from Riga"],
}

# East-Asian names are written surname-first: "Ding Liren", not "Liren Ding"
SURNAME_FIRST = {"Ding, Liren", "Wei, Yi", "Yu, Yangyi", "Wang, Hao", "Bu, Xiangzhi",
                 "Lu, Shanglei", "Xu, Xiangyu", "Le, Quang Liem"}

# FIDE names whose "Surname, First" split doesn't give a name anyone types
# (Indian given-name/initial conventions, double surnames): PLAYER_ALIASES replaces
# the derived forms entirely, its first entry being the canonical full name.
NO_DERIVE = {"Vidit, Santosh Gujrathi", "Pranav, V", "Aravindh, Chithambaram VR.",
             "Maroroa Jones, Gawain C B", "Erigaisi Arjun", "Tabatabaei, M. Amin"}

# Username -> FIDE name registry from src/crawl_top_players.py
# ({fide_id: {"name": "Carlsen, Magnus", "chesscom": [...], "lichess": [...]}}).
PLAYER_REGISTRY = Path(__file__).resolve().parent.parent / "data/top_players/accounts.json"


def _derived_names(fide: str) -> list[str]:
    """"Carlsen, Magnus" -> ["Magnus Carlsen", "Carlsen"]; "Gukesh D" -> ["Gukesh"]."""
    if fide in NO_DERIVE:
        return list(PLAYER_ALIASES[fide])
    if "," not in fide:
        return [re.sub(r"(?:\s+[A-Z]\.?)+$", "", fide).strip()]
    last, first = (x.strip() for x in fide.split(",", 1))
    # trailing initials: "Haik M." -> "Haik", "David W L" -> "David", "Mikhail Al." -> "Mikhail"
    first = re.sub(r"(?:\s+(?:[A-Z]{1,2}\.?|Al\.))+$", "", first)
    last = re.sub(r"^(Van|De|Von|Der|Den) ", lambda m: m[1].lower() + " ", last)  # "van Foreest"
    if fide in SURNAME_FIRST:
        return [f"{last} {first}"]
    # a bare 2-3 letter surname ("So") is an ordinary word; keep only the full name
    return [f"{first} {last}"] + ([last] if len(last) > 3 else [])


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


@lru_cache(maxsize=1)
def _fide_index() -> dict[str, str]:
    """lowercase FIDE name / natural full name / site username -> FIDE name."""
    fide_names = set(PLAYER_ALIASES)
    usernames = {}
    if PLAYER_REGISTRY.exists():
        for p in json.loads(PLAYER_REGISTRY.read_text()).values():
            fide_names.add(p["name"])
            for u in p.get("chesscom", []) + p.get("lichess", []):
                usernames[_norm(u)] = p["name"]
    index = {}
    for f in fide_names:
        index[_norm(f)] = f
        index.setdefault(_norm(_derived_names(f)[0]), f)
    for u, f in usernames.items():
        index.setdefault(u, f)
    for f in PLAYER_ALIASES:  # bare surname lookup ("Tal", "Kasparov") for listed players
        index.setdefault(_norm(f.split(",")[0]), f)
    return index


def player_names(full: str) -> list[str]:
    """Name variants a user might type for this player. `full` may be a FIDE/PGN
    name ("Carlsen, Magnus"), a plain name, or a Chess.com/Lichess username."""
    fide = _fide_index().get(_norm(full))
    if fide is None and "," in full:
        fide = full.strip()  # unknown PGN-style name: still derive "First Last"
    if fide is None:
        full = " ".join(full.split())
        sur = full.split()[-1] if full else full
        return [full, sur] if sur != full else [full]
    out, seen = [], set()
    for n in _derived_names(fide) + PLAYER_ALIASES.get(fide, []):
        if n and n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def render_player(name: str, rng: random.Random) -> str:
    return rng.choice(PLAYER).format(p=rng.choice(player_names(name)))


ATTRS = ("elo", "time_control", "opening", "captures")


def render(m: dict, rng: random.Random, capture_thresh=DEFAULT_CAPTURE_THRESHOLDS) -> list[str]:
    """One label per attribute, ATTRS order."""
    return [render_elo(m, rng), render_tc(m, rng), render_opening(m, rng),
            render_captures(m, rng, capture_thresh)]
