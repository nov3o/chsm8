"""Player prompts: "play like Magnus", "you are Tal", ... for the player games in SFT.

Each player game gets one prompt drawn from PROMPTS with one of the player's names (full name, surname, nicknames).

    render("Magnus Carlsen", random.Random(0))   # e.g. "channel Magnus"
"""
import random

PROMPTS = [
    # bare name
    "{p}",
    # commands
    "play like {p}", "play like {p} would", "play in the style of {p}", "play {p}-style chess",
    "make moves like {p}", "copy {p}'s style", "imitate {p}", "channel {p}",
    "do your best {p} impression", "give me your {p}", "{p} mode on", "{p} style please",
    # personas
    "you are {p}", "you're {p} today", "pretend you're {p}", "be {p} for this game",
    "play as if you were {p}", "imagine you are {p} at the board", "take the seat of {p}",
    "you're {p}, show me how it's done", "become {p}", "think like {p}",
    # questions
    "how would {p} play this?", "can you play like {p}?",
    "could you play the way {p} does?", "show me how {p} plays", "let me see some {p} chess",
    "can I get a {p} game?", "i want to play against {p}",
    # casual
    "gimme {p} vibes", "full {p} energy", "play like prime {p}", "go {p} on me",
    "i'm a big {p} fan, play like that", "play like my favourite player, {p}",
    "channel your inner {p}", "{p} would be proud, play like that", "bring out the {p} in you",
    "just be {p}",
    # framing
    "play the way {p} plays over the board", "moves {p} would make", "{p}'s way of playing",
    "play with {p}'s ideas", "follow {p}'s approach", "play it the {p} way",
    "same style as {p}", "like {p} at their best", "sound like {p} on the board",
]

# Names beyond "First Last" and "Last", keyed by full name. Only well-established nicknames: a wrong alias teaches
# the model a wrong association.
ALIASES = {
    "Magnus Carlsen": ["Magnus"],
    "Bobby Fischer": ["Bobby", "Fischer", "Robert James Fischer"],
    "Garry Kasparov": ["Garry", "Gazza"],
    "Anatoly Karpov": ["Tolya"],
    "Judit Polgar": ["Judit"],
    "Hikaru Nakamura": ["Hikaru", "Naka"],
    "Fabiano Caruana": ["Fabiano", "Fabi"],
    "Javokhir Sindarov": ["Javokhir"],
    "Nodirbek Abdusattorov": ["Nodirbek"],
    "Wesley So": ["Wesley"],
    "Praggnanandhaa Rameshbabu": ["Pragg", "Praggnanandhaa"],
    "Vincent Keymer": ["Vincent"],
    "Alireza Firouzja": ["Alireza"],
    "Anish Giri": ["Anish"],
    "Arjun Erigaisi": ["Arjun"],
    "Jan-Krzysztof Duda": ["JKD", "JK Duda", "Jan-Krzysztof"],
    "Viswanathan Anand": ["Vishy", "Vishy Anand", "Tiger of Madras"],
    "Levon Aronian": ["Levon"],
    "Ding Liren": ["Ding"],
    "Ian Nepomniachtchi": ["Nepo", "Ian"],
    "Hans Niemann": ["Hans"],
    "Maxime Vachier-Lagrave": ["MVL", "Maxime"],
    "Shakhriyar Mamedyarov": ["Shakh", "Shakhriyar"],
    "Veselin Topalov": ["Veselin", "Topa"],
    "Vasyl Ivanchuk": ["Chucky"],
    "Mikhail Tal": ["the Magician from Riga"],
}
SURNAME_FIRST = {"Ding Liren", "Wei Yi", "Yu Yangyi", "Wang Hao", "Bu Xiangzhi", "Le Quang Liem"}


def names(full):
    """The ways a user might name this player: the full name, the surname if it is not a short common word, aliases."""
    parts = full.split()
    surname = parts[0] if full in SURNAME_FIRST else parts[-1]
    out = [full] + ([surname] if len(parts) > 1 and len(surname) > 3 and full not in SURNAME_FIRST else [])
    for a in ALIASES.get(full, []):
        if a.lower() not in {n.lower() for n in out}:
            out.append(a)
    return out


def render(full, rng=random):
    return rng.choice(PROMPTS).format(p=rng.choice(names(full)))
