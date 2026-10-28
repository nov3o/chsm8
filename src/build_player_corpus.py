#!/usr/bin/env python3
"""Top-player games -> SFT corpus in the data/sft format (rows.i16, legal.bin, index.parquet).

Input: data/top_players/games/<fide_id>.jsonl (one record per (player, game); moves in LAN,
e.g. "Ng1-f3", "e7-e8=Q", "O-O"; a game between two top-100 players appears once per side).
One index row per record, focal side = player_side. No stored labels: at train time the
trainer draws 1-2 of {render_player (meta_templates.player_names), elo, tc, opening, captures}.
Split: blake2b of the game url (90/5/5), so both sides of a shared game land in the same split.
Opening: deepest match in the Lichess chess-openings table (data/openings/*.tsv, CC0),
the same naming as the Lichess `opening` column in data/sft.
usage: python src/build_player_corpus.py --out data/sft_players
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from build_sft_corpus import to_codes  # noqa: E402
from build_tier3 import LEGAL_DTYPE, replay_game  # noqa: E402

OPENINGS: dict[tuple, tuple[str, str]] = {}


def load_openings(dirpath: Path) -> dict[tuple, tuple[str, str]]:
    import chess
    table = {}
    for f in sorted(dirpath.glob("*.tsv")):
        for line in f.read_text().splitlines()[1:]:
            eco, name, pgn = line.split("\t")
            board, key = chess.Board(), []
            for tok in pgn.split():
                if tok[0].isdigit():
                    continue
                key.append(board.push_san(tok).uci())
            table[tuple(key)] = (eco, name)
    return table


def lan_to_uci(moves: list[str]) -> list[str]:
    """LAN -> UCI via the board (castling "O-O" needs the side to move)."""
    import chess
    board, out = chess.Board(), []
    for m in moves:
        t = m.rstrip("+#")
        if t.startswith("O-O"):
            mv = board.parse_san(t)
        else:
            body, _, promo = t.partition("=")
            body = body.lstrip("KQRBN").replace("x", "-")
            src, dst = body.split("-")
            mv = chess.Move.from_uci(src + dst + promo.lower())
        if mv not in board.legal_moves:
            raise ValueError(f"illegal {m}")
        out.append(mv.uci()); board.push(mv)
    return out


def opening_of(uci: list[str]) -> tuple[str, str] | tuple[None, None]:
    best = (None, None)
    for n in range(1, min(len(uci), 40) + 1):
        hit = OPENINGS.get(tuple(uci[:n]))
        if hit:
            best = hit
    return best


def parse_tc(tc: str) -> tuple[int | None, int | None]:
    try:
        base, _, inc = str(tc).partition("+")
        return int(base), int(inc or 0)
    except ValueError:  # daily games ("1/86400") etc.
        return None, None


def split_of(url: str, seed: int) -> str:
    v = int.from_bytes(hashlib.blake2b(f"{seed}:{url}".encode(), digest_size=8).digest(), "little") % 100
    return "train" if v < 90 else ("val" if v < 95 else "test")


def process(job):
    path, tmp, max_rows, seed = job
    names = json.loads((ROOT / "data/top_players/accounts.json").read_text())
    rows_p, legal_p = f"{tmp}/{Path(path).stem}.rows", f"{tmp}/{Path(path).stem}.legal"
    metas, cum, skipped = [], 0, {}
    with open(rows_p, "wb") as fr, open(legal_p, "wb") as fl:
        for i, line in enumerate(open(path)):
            r = json.loads(line)
            try:
                uci = lan_to_uci(r["moves"])
                codes, stats = to_codes(uci)
                rows, legal = replay_game(codes)
            except Exception as e:  # noqa: BLE001
                skipped[type(e).__name__] = skipped.get(type(e).__name__, 0) + 1
                continue
            side = r["player_side"]
            n, caps, checks = stats[side]
            if n == 0 or len(rows) > max_rows:
                skipped["len"] = skipped.get("len", 0) + 1
                continue
            fr.write(np.asarray(rows, dtype="<i2").tobytes())
            fl.write(np.array(legal, dtype=LEGAL_DTYPE).tobytes())
            opp = "black" if side == "white" else "white"
            base, inc = parse_tc(r["tc"])
            eco, opening = opening_of(uci)
            fid = str(r["player_fide_id"])
            metas.append({
                "month": "top100", "row_idx": i, "side": side, "player_fide_id": fid,
                "player": names.get(fid, {}).get("name", r[side]), "account": r[side],
                "opponent": r[opp], "player_elo": int(r[f"{side}_elo"] or 0),
                "opponent_elo": int(r[f"{opp}_elo"] or 0), "result": r["result"], "date": r["date"],
                "source": r["source"], "url": r["url"], "time_class": r["time_class"],
                "time_control": r["tc"], "tc_base": base, "tc_increment": inc,
                "eco": eco, "opening": opening, "n_plies": len(uci),
                "side_moves": n, "captures": caps, "checks": checks,
                "split": split_of(r["url"], seed), "row_off": cum, "n_rows": len(rows)})
            cum += len(rows)
    return {"path": path, "rows": rows_p, "legal": legal_p, "meta": metas, "n_rows": cum, "skipped": skipped}


def init(openings_dir):
    global OPENINGS
    OPENINGS = load_openings(Path(openings_dir))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", default=str(ROOT / "data/top_players/games"))
    ap.add_argument("--openings", default=str(ROOT / "data/openings"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    ap.add_argument("--max-rows", type=int, default=512)
    ap.add_argument("--split-seed", type=int, default=20261001)
    a = ap.parse_args()
    out = Path(a.out); tmp = out / "_tmp"; tmp.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.games).glob("*.jsonl"))
    metas, offset, skipped = [], 0, {}
    with open(out / "rows.i16", "wb") as fr, open(out / "legal.bin", "wb") as fl, \
            mp.Pool(a.workers, initializer=init, initargs=(a.openings,)) as pool:
        for part in pool.imap(process, [(str(f), str(tmp), a.max_rows, a.split_seed) for f in files]):
            for k, h in (("rows", fr), ("legal", fl)):
                with open(part[k], "rb") as src:
                    while chunk := src.read(64 << 20):
                        h.write(chunk)
                os.remove(part[k])
            for m in part["meta"]:
                m["row_off"] += offset
            metas.extend(part["meta"]); offset += part["n_rows"]
            for k, v in part["skipped"].items():
                skipped[k] = skipped.get(k, 0) + v
            print(f"[players] {Path(part['path']).name}: {len(part['meta'])} kept (total {len(metas)})", flush=True)
    pq.write_table(pa.Table.from_pylist(metas), out / "index.parquet")
    tmp.rmdir()
    summary = {"records": len(metas), "rows": offset, "skipped": skipped,
               "players": len({m["player_fide_id"] for m in metas}),
               "splits": {s: sum(m["split"] == s for m in metas) for s in ("train", "val", "test")},
               "opening_found": round(sum(m["opening"] is not None for m in metas) / max(len(metas), 1), 4),
               "tc_parsed": round(sum(m["tc_base"] is not None for m in metas) / max(len(metas), 1), 4)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
