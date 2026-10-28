#!/usr/bin/env python3
"""
crawl_top_players.py — up to N (default 10k) games per FIDE top-100 player,
newest first, from open sources. Replaces the YottaChess scrape (yottachess.com
sits behind a Cloudflare challenge + reCAPTCHA-gated downloads).

Sources:
  - FIDE top-100 list (ratings.fide.com)  -> names, FIDE IDs, ratings
  - Chess.com public API (api.chess.com)   -> online games of every account of
                                             the player; accounts are found by
                                             real-name match on verified GM
                                             profiles (/pub/titled/GM)
  - Lichess API (lichess.org/api)          -> online games; accounts only from
                                             the accounts file (Lichess has no
                                             verified real-name lookup)
  - TWIC (theweekinchess.com), optional    -> OTB games matched on FIDE id
                                             (--twic-issues N, default off)

Accounts file (--accounts, JSON) is both input and output: auto-discovered
Chess.com matches are merged into it, so it can be reviewed/corrected by hand
and the next run uses the corrected mapping:
  {"<fide_id>": {"name": "...", "chesscom": ["hikaru"], "lichess": ["Hikaru"]}}
An entry with "locked": true is never touched by auto-discovery.

Selection policy (players with more than --per-player games):
  1. Window: each account is read newest month first and stops once it has
     >= --per-player games of ANY speed (bullet counts). The window is
     count-based, not a date range: a few months for very active accounts,
     years (or the whole history) for others. Older games are never seen.
  2. Pool: all of the player's accounts (+ TWIC if enabled), deduplicated.
  3. Rank: time-class tier — classical/daily > rapid > blitz > bullet —
     then newest first; keep the top --per-player.
  So bullet is dropped first *within the recent window*, but a player whose
  recent play is mostly bullet still ends up mostly bullet (e.g. Rapport,
  Martirosyan, Erigaisi >60%). Players with fewer games keep everything.
  This is intentional: the corpus is meant to reflect each player's *recent*
  style. Filter on "time_class" downstream if bullet is unwanted.

Output: one JSONL shard per player, <root>/games/<fide_id>.jsonl (finished
shards are skipped on rerun; --refresh to redo), then a merged file --out. Records use the extract_lan_moves.py
LAN "moves" format plus metadata:
  {"white","black","white_elo","black_elo","result","date","event","tc",
   "source","account","time_class","url","player_fide_id","player_side",
   "moves":[LAN...]}
The merged file has one record per (player, game): a game between two top-100
players appears twice, once per player_side (white/black), so each player's
side can be trained with that player's prompt.
(--keep-pgn adds the raw PGN; off by default — 1M PGNs is ~1.5 GB.)
"""
import argparse
import io
import json
import os
import re
import sys
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.pgn
import requests

UA = "llm_chess-research-crawler (contact: ilya.badaneen@gmail.com)"
FIDE_TOP = "https://ratings.fide.com/a_top.php?list=open"
CC = "https://api.chess.com/pub"
LI = "https://lichess.org/api"
TWIC_INDEX = "https://theweekinchess.com/twic"

TIER = {"classical": 0, "daily": 0, "correspondence": 0, "rapid": 1, "blitz": 2, "bullet": 3}

session = requests.Session()
session.headers["User-Agent"] = UA


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def fetch(url, cache: Path | None, binary=False, headers=None, retries=5):
    """GET with optional on-disk cache. Returns bytes/str, or None on 404/410."""
    path = None
    if cache is not None:
        path = cache / re.sub(r"[^A-Za-z0-9._-]", "_", url.split("://", 1)[1])
        if path.exists():
            return path.read_bytes() if binary else path.read_text()
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=300, headers=headers)
        except requests.RequestException as e:
            log(f"  retry {url}: {e}")
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (404, 410):
            return None
        if r.status_code == 429:  # both APIs ask for a ~1 min back-off
            log(f"  429 on {url}, sleeping 60s")
            time.sleep(60)
            continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        if path is not None:
            path.write_bytes(r.content)
        return r.content if binary else r.text
    raise RuntimeError(f"failed after {retries} tries: {url}")


def norm_name(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return frozenset(t for t in re.split(r"[^a-z]+", s.lower()) if len(t) > 1)


# ---------------------------------------------------------------- parsing
def tc_class(tc):
    """Classify a PGN TimeControl ("600+5", "40/5400:1800+30") by estimated
    duration (base + 40*increment), same buckets as Lichess."""
    m = re.match(r"(?:\d+/)?(\d+)", tc or "")
    if not m:
        return None
    inc = re.search(r"\+(\d+)", tc)
    est = int(m[1]) + 40 * (int(inc[1]) if inc else 0)
    return "bullet" if est < 180 else "blitz" if est < 480 else "rapid" if est < 1500 else "classical"


def to_record(game, keep_pgn, **extra):
    """PGN game -> record with LAN moves, or None if unusable."""
    h = game.headers
    if h.get("Variant", "Standard").lower() not in ("standard", "") or "FEN" in h:
        return None
    if h.get("Result") not in ("1-0", "0-1", "1/2-1/2"):
        return None
    board = game.board()
    moves = []
    for mv in game.mainline_moves():
        moves.append(board.lan(mv))
        board.push(mv)
    if len(moves) < 10:
        return None

    def elo(k):
        try:
            return int(h.get(k, ""))
        except ValueError:
            return None
    date = h.get("UTCDate") or h.get("Date", "")
    rec = {"white": h.get("White"), "black": h.get("Black"),
           "white_elo": elo("WhiteElo"), "black_elo": elo("BlackElo"),
           "result": h["Result"], "date": date.replace(".", "-"),
           "event": h.get("Event"), "tc": h.get("TimeControl")}
    rec.update(extra)
    rec["moves"] = moves
    if keep_pgn:
        rec["pgn"] = game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False))
    return rec


def read_pgn(text):
    g = chess.pgn.read_game(io.StringIO(text))
    return g if g is not None and not g.errors else None


# ---------------------------------------------------------------- FIDE
def fide_top(cache, n):
    html = fetch(FIDE_TOP, cache)
    out = [{"fide_id": m[1], "name": m[2].strip(), "rating": int(m[3])}
           for m in re.finditer(r'href=/profile/(\d+)>([^<]+)</a>.*?rating_column>(\d+)<', html, re.S)]
    return out[:n]


# ---------------------------------------------------------------- accounts
def discover_chesscom(players, accounts, cache):
    """Add every verified-GM Chess.com account whose real name matches."""
    want = {norm_name(p["name"]): p["fide_id"] for p in players}
    titled = json.loads(fetch(f"{CC}/titled/GM", cache))["players"]
    log(f"Chess.com: scanning {len(titled)} verified GM profiles for real-name matches")
    hits = defaultdict(set)
    for i, u in enumerate(titled):
        raw = fetch(f"{CC}/player/{u}", cache)
        if raw:
            key = norm_name(json.loads(raw).get("name", ""))
            if key in want:
                hits[want[key]].add(u)
        if i % 500 == 499:
            log(f"  {i + 1}/{len(titled)} profiles, {len(hits)} players matched")
    for p in players:
        a = accounts.setdefault(p["fide_id"], {"name": p["name"], "chesscom": [], "lichess": []})
        if not a.get("locked"):
            a["chesscom"] = sorted(set(a.get("chesscom", [])) | hits[p["fide_id"]])


# ---------------------------------------------------------------- Chess.com
def chesscom_games(user, need, cache, keep_pgn, no_bullet):
    raw = fetch(f"{CC}/player/{user}/games/archives", None)
    if raw is None:
        log(f"  chess.com {user}: not found")
        return []
    archives = json.loads(raw)["archives"][::-1]  # newest month first
    out = []
    for i, url in enumerate(archives):
        # the two newest months can still change -> don't cache them
        month = fetch(url, cache if i >= 2 else None)
        for g in json.loads(month or "{}").get("games", [])[::-1]:
            tcl = g.get("time_class")
            if g.get("rules") != "chess" or not g.get("pgn") or (no_bullet and tcl == "bullet"):
                continue
            pg = read_pgn(g["pgn"])
            if pg is None:
                continue
            rec = to_record(pg, keep_pgn, source="chess.com", account=user, time_class=tcl, url=g.get("url"))
            if rec:
                rec["white_elo"], rec["black_elo"] = g["white"].get("rating"), g["black"].get("rating")
                out.append(rec)
        if len(out) >= need:
            break
    return out


# ---------------------------------------------------------------- Lichess
def lichess_games(user, need, keep_pgn, no_bullet, token=None):
    perfs = ["blitz", "rapid", "classical", "correspondence"]
    if not no_bullet:
        perfs = ["ultraBullet", "bullet"] + perfs
    url = (f"{LI}/games/user/{user}?max={need}&rated=true&perfType={','.join(perfs)}"
           "&moves=true&tags=true&clocks=false&evals=false&opening=false")
    hdr = {"Accept": "application/x-chess-pgn"}
    if token:
        hdr["Authorization"] = f"Bearer {token}"
    text = fetch(url, None, headers=hdr)
    if text is None:
        log(f"  lichess {user}: not found")
        return []
    out = []
    for chunk in re.split(r"\n\n(?=\[Event )", text):
        pg = read_pgn(chunk)
        if pg is None:
            continue
        ev = pg.headers.get("Event", "").lower()
        tcl = next((w for w in ("bullet", "blitz", "rapid", "classical", "correspondence") if w in ev),
                   None) or tc_class(pg.headers.get("TimeControl"))
        rec = to_record(pg, keep_pgn, source="lichess", account=user, time_class=tcl,
                        url=pg.headers.get("Site"))
        if rec:
            out.append(rec)
    return out


# ---------------------------------------------------------------- TWIC (optional OTB)
def twic_games(players, cache, n_issues, keep_pgn):
    ids = {p["fide_id"] for p in players}
    html = fetch(TWIC_INDEX, cache)
    nums = sorted({int(n) for n in re.findall(r"zips/twic(\d+)g\.zip", html)}, reverse=True)[:n_issues]
    log(f"TWIC: {len(nums)} issues")
    by_player = defaultdict(list)

    def one(n):
        blob = fetch(f"https://theweekinchess.com/zips/twic{n}g.zip", cache, binary=True)
        if blob is None:
            return []
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
        res = []
        for chunk in re.split(r"\n(?=\[Event )", text):
            hit = set(re.findall(r'\[(?:White|Black)FideId "(\d+)"\]', chunk)) & ids
            if not hit:
                continue
            pg = read_pgn(chunk)
            if pg is None:
                continue
            site = pg.headers.get("Site", "").lower()
            if "chess.com" in site or "lichess" in site or site.endswith(" int"):
                continue  # online event reprinted by TWIC; the APIs cover those
            ev = pg.headers.get("Event", "").lower()
            tcl = next((w for w in ("bullet", "blitz", "rapid") if w in ev), None) \
                or tc_class(pg.headers.get("TimeControl")) or "classical"
            rec = to_record(pg, keep_pgn, source=f"twic{n}", account="otb", time_class=tcl,
                            white_fide_id=pg.headers.get("WhiteFideId"),
                            black_fide_id=pg.headers.get("BlackFideId"))
            if rec:
                res.append((hit, rec))
        return res

    with ThreadPoolExecutor(8) as ex:
        for got in ex.map(one, nums):
            for hit, rec in got:
                for fid in hit:
                    by_player[fid].append(rec)
    return by_player


# ---------------------------------------------------------------- main
def game_key(r):
    # Online games have a unique URL. date+moves alone is NOT unique: forced
    # repetition lines get played by many players on the same day. It's only the
    # fallback for TWIC/OTB games (no URL), where names differ across sources.
    if r.get("url"):
        return r["url"].split("?")[0].rstrip("/").lower()
    return r["date"], " ".join(r["moves"])


def player_side(rec, fid):
    """Which colour the crawled player had: the fetched account (online) or the
    FIDE id (TWIC) is matched against the game's White/Black."""
    if rec.get("white_fide_id") == fid:
        return "white"
    if rec.get("black_fide_id") == fid:
        return "black"
    acc = (rec.get("account") or "").lower()
    if (rec.get("white") or "").lower() == acc:
        return "white"
    if (rec.get("black") or "").lower() == acc:
        return "black"
    return None


def select(pool, n):
    uniq = {game_key(r): r for r in pool}.values()
    ranked = sorted(uniq, key=lambda r: r["date"], reverse=True)
    ranked.sort(key=lambda r: TIER.get(r["time_class"], 9))  # stable: tier, then newest
    return ranked[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/top_players")
    ap.add_argument("--out", default=None, help="merged JSONL (default <root>/top100_games.jsonl)")
    ap.add_argument("--accounts", default=None, help="accounts JSON (default <root>/accounts.json)")
    ap.add_argument("--per-player", type=int, default=10000)
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--only", nargs="*", help="restrict to these FIDE ids")
    ap.add_argument("--no-bullet", action="store_true", help="drop bullet/ultrabullet")
    ap.add_argument("--twic-issues", type=int, default=0, help="also scan N recent TWIC weeks for OTB games")
    ap.add_argument("--skip-discovery", action="store_true", help="use accounts file as-is")
    ap.add_argument("--refresh", action="store_true", help="recrawl players with an existing shard")
    ap.add_argument("--keep-pgn", action="store_true")
    ap.add_argument("--lichess-token", default=None, help="optional; raises Lichess export rate")
    args = ap.parse_args()

    root = Path(args.root)
    cache, shards = root / "cache", root / "games"
    cache.mkdir(parents=True, exist_ok=True)
    shards.mkdir(parents=True, exist_ok=True)
    acc_path = Path(args.accounts or root / "accounts.json")
    out_path = Path(args.out or root / "top100_games.jsonl")

    players = fide_top(cache, args.top)
    if args.only:
        players = [p for p in players if p["fide_id"] in set(args.only)]
    log(f"FIDE top: {len(players)} players")

    accounts = json.loads(acc_path.read_text()) if acc_path.exists() else {}
    if not args.skip_discovery:
        discover_chesscom(players, accounts, cache)
    for p in players:
        accounts.setdefault(p["fide_id"], {"name": p["name"], "chesscom": [], "lichess": []})
    acc_path.write_text(json.dumps(accounts, indent=1, ensure_ascii=False))
    log(f"accounts -> {acc_path}: "
        f"{sum(bool(accounts[p['fide_id']]['chesscom']) for p in players)} with chess.com, "
        f"{sum(bool(accounts[p['fide_id']].get('lichess')) for p in players)} with lichess")

    otb = twic_games(players, cache, args.twic_issues, args.keep_pgn) if args.twic_issues else {}

    summary = []
    for i, p in enumerate(players):
        fid, a = p["fide_id"], accounts[p["fide_id"]]
        shard = shards / f"{fid}.jsonl"
        if shard.exists() and not args.refresh:
            summary.append({**p, "games": sum(1 for _ in shard.open()), "skipped": True})
            continue
        pool = list(otb.get(fid, []))
        for u in a.get("chesscom", []):
            got = chesscom_games(u, args.per_player, cache, args.keep_pgn, args.no_bullet)
            log(f"  chess.com {u}: {len(got)}")
            pool += got
        for u in a.get("lichess", []):
            got = lichess_games(u, args.per_player, args.keep_pgn, args.no_bullet, args.lichess_token)
            log(f"  lichess {u}: {len(got)}")
            pool += got
        chosen = select(pool, args.per_player)
        for r in chosen:
            r["player_fide_id"] = fid
            r["player_side"] = player_side(r, fid)
        tmp = shard.with_suffix(f".{os.getpid()}.tmp")  # per-process: concurrent runs can't collide
        with tmp.open("w") as f:
            for r in chosen:
                f.write(json.dumps(r) + "\n")
        tmp.rename(shard)  # atomic: a shard exists only when complete
        tc = Counter(r["time_class"] for r in chosen)
        summary.append({**p, "games": len(chosen), "time_class": dict(tc)})
        log(f"[{i + 1}/{len(players)}] {p['name']}: {len(chosen)} games {dict(tc)}")

    # merge: one record per (player, game). A game between two top-100 players is
    # kept once per side — each is a separate training example (that player's moves,
    # that player's prompt). Duplicates within a player were already removed in select().
    n, games, no_side = 0, set(), 0
    with out_path.open("w") as f:
        for p in players:
            shard = shards / f"{p['fide_id']}.jsonl"
            if not shard.exists():
                continue
            for line in shard.open():
                r = json.loads(line)
                if "player_side" not in r:  # shards written before the field existed
                    r["player_side"] = player_side(r, p["fide_id"])
                no_side += r["player_side"] is None
                games.add(game_key(r))
                f.write(json.dumps(r) + "\n")
                n += 1
    (root / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    short = [f"{s['name']}={s['games']}" for s in summary if s["games"] < args.per_player]
    log(f"merged {n} (player, game) records, {len(games)} unique games -> {out_path}"
        + (f"; {no_side} records with unknown player_side" if no_side else ""))
    log(f"{len(short)}/{len(summary)} players under {args.per_player}: {', '.join(short)}")


if __name__ == "__main__":
    main()
