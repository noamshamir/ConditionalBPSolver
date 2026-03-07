"""Utility functions — puz conversion, grid printing, clue preprocessing.

Closely matches Berkeley-Crossword-Solver utilities so that evaluation is
directly comparable.
"""
from __future__ import annotations

import math
import re
import string
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import puz as puzlib


# ======================================================================
# .puz → JSON conversion  (matches Berkeley's utils.puz_to_json)
# ======================================================================
def puz_to_json(fname: str | Path) -> dict:
    """Convert a ``.puz`` file to the JSON dict expected by ``Crossword``."""
    p = puzlib.read(str(fname))
    numbering = p.clue_numbering()

    grid = [[None for _ in range(p.width)] for _ in range(p.height)]
    for row_idx in range(p.height):
        cell = row_idx * p.width
        row_solution = p.solution[cell : cell + p.width]
        for col_idx, _ in enumerate(row_solution):
            if p.solution[cell + col_idx : cell + col_idx + 1] == ".":
                grid[row_idx][col_idx] = "BLACK"
            else:
                grid[row_idx][col_idx] = ["", row_solution[col_idx : col_idx + 1]]

    across_clues: Dict[str, list] = {}
    for clue in numbering.across:
        answer = "".join(p.solution[clue["cell"] + i] for i in range(clue["len"]))
        across_clues[str(clue["num"])] = [clue["clue"] + " ", " " + answer]
        r, c = divmod(clue["cell"], p.width)
        grid[r][c][0] = str(clue["num"])

    down_clues: Dict[str, list] = {}
    for clue in numbering.down:
        answer = "".join(
            p.solution[clue["cell"] + i * numbering.width]
            for i in range(clue["len"])
        )
        down_clues[str(clue["num"])] = [clue["clue"] + " ", " " + answer]
        r, c = divmod(clue["cell"], p.width)
        grid[r][c][0] = str(clue["num"])

    return {
        "metadata": {"date": None, "rows": p.height, "cols": p.width},
        "clues": {"across": across_clues, "down": down_clues},
        "grid": grid,
    }


def puz_to_pairs(fname: str | Path) -> List[Tuple[str, str]]:
    """Return ``[(clue, ANSWER), …]`` from a ``.puz`` file."""
    p = puzlib.read(str(fname))
    numbering = p.clue_numbering()
    pairs: Dict[str, str] = {}
    for clue in numbering.across:
        answer = "".join(p.solution[clue["cell"] + i] for i in range(clue["len"]))
        pairs[clue["clue"]] = answer
    for clue in numbering.down:
        answer = "".join(
            p.solution[clue["cell"] + i * numbering.width]
            for i in range(clue["len"])
        )
        pairs[clue["clue"]] = answer
    return list(pairs.items())


# ======================================================================
# Clue preprocessing  (matches Berkeley's models.preprocess_clue_fn)
# ======================================================================
def preprocess_clue(clue: str) -> str:
    """Normalise a crossword clue string."""
    clue = str(clue)
    # strip accents
    clue = "".join(
        c
        for c in unicodedata.normalize("NFD", clue)
        if unicodedata.category(c) != "Mn"
    )

    clue = re.sub("\x17|\x18|\x93|\x94|\u201c|\u201d|''|\"\"", '"', clue)
    clue = re.sub("\x85|\u2026", "...", clue)
    clue = re.sub("\x91|\x92|\u2018|\u2019", "'", clue)
    clue = re.sub("\u201a", ",", clue)
    clue = re.sub("\u2014|\u2013", "-", clue)
    clue = re.sub("\u00a2", " cents", clue)
    clue = re.sub("\u00bf|\u00a1|^;|\\{|\\}", "", clue)
    clue = re.sub("\u00f7", "division", clue)
    clue = re.sub("\u00b0", " degrees", clue)

    # currency
    euro = re.search(r"^£[0-9]+(,*[0-9]*){0,}| £[0-9]+(,*[0-9]*){0,}", clue)
    if euro:
        num = clue[: euro.end()]
        rest = clue[euro.end() :]
        clue = num + " Euros" + rest
        clue = re.sub(", Euros", " Euros", clue)
        clue = re.sub("Euros [Mm]illion", "million Euros", clue)
        clue = re.sub("Euros [Bb]illion", "billion Euros", clue)
        clue = re.sub("Euros[Kk]", "K Euros", clue)
        clue = re.sub(" K Euros", "K Euros", clue)
        clue = re.sub("£", "", clue)

    # strip trailing enumerations like (5) or (3, 5)
    clue = re.sub(r" *\(\d{1,},*\)$| *\(\d{1,},* \d{1,}\)$", "", clue)

    clue = re.sub("&amp;", "&", clue)
    clue = re.sub("&lt;", "<", clue)
    clue = re.sub("&gt;", ">", clue)

    # common abbreviation expansions
    clue = re.sub(r"e\.g\.|for ex\.", "for example", clue)
    clue = re.sub(
        r": [Aa]bbreviat\.|: [Aa]bbrev\.|: [Aa]bbrv\.|: [Aa]bbrv|: [Aa]bbr\.|: [Aa]bbr",
        " abbreviation",
        clue,
    )
    clue = re.sub(r"abbr\.|abbrv\.", "abbreviation", clue)
    clue = re.sub(r"Abbr\.|Abbrv\.", "Abbreviation", clue)
    clue = re.sub(r"\(anag\.\)|\(anag\)", "(anagram)", clue)
    clue = re.sub(r"org\.", "organization", clue)
    clue = re.sub(r"Org\.", "Organization", clue)
    clue = re.sub(r"Grp\.|Gp\.", "Group", clue)
    clue = re.sub(r"grp\.|gp\.", "group", clue)
    clue = re.sub(r": Sp\.", " (Spanish)", clue)
    clue = re.sub(r"\(Sp\.\)|Sp\.", "(Spanish)", clue)
    clue = re.sub(r"Ave\.", "Avenue", clue)
    clue = re.sub(r"Sch\.", "School", clue)
    clue = re.sub(r"sch\.", "school", clue)
    clue = re.sub(r"Agcy\.", "Agency", clue)
    clue = re.sub(r"agcy\.", "agency", clue)
    clue = re.sub(r"Co\.", "Company", clue)
    clue = re.sub(r"co\.", "company", clue)
    clue = re.sub(r"No\.", "Number", clue)
    clue = re.sub(r"no\.", "number", clue)
    clue = re.sub(r": [Vv]ar\.", " variable", clue)
    clue = re.sub(r"Subj\.", "Subject", clue)
    clue = re.sub(r"subj\.", "subject", clue)
    clue = re.sub(r"Subjs\.", "Subjects", clue)
    clue = re.sub(r"subjs\.", "subjects", clue)

    # theme clues
    theme_clue = re.search(r"^.+\|[A-Z]{1,}", clue)
    if theme_clue:
        clue = re.sub(r"\|", " | ", clue)

    if "Partner of" in clue:
        clue = re.sub("Partner of", "", clue)
        clue = clue + " and ___"

    link = re.search(r"^.+-.+ [Ll]ink$", clue)
    if link:
        no_link = re.search(r"^.+-.+ ", clue)
        x_y = clue[no_link.start() : no_link.end() - 1]
        parts = x_y.split("-")
        clue = parts[0] + " ___ " + parts[1]

    follower = re.search(r"^.+ [Ff]ollower$", clue)
    if follower:
        no_f = re.search(r"^.+ ", clue)
        clue = clue[: no_f.end() - 1] + " ___"

    preceder = re.search(r"^.+ [Pp]receder$", clue)
    if preceder:
        no_p = re.search(r"^.+ ", clue)
        clue = "___ " + clue[: no_p.end() - 1]

    if re.search(r"--[^A-Za-z]|--$", clue):
        clue = re.sub("--", "__", clue)
    if not re.search(r"_-[A-Za-z]|_-$", clue):
        clue = re.sub("_-", "__", clue)

    clue = re.sub(r"_{2,}", "___", clue)
    clue = re.sub(r"\?$", " (wordplay)", clue)

    nonverbal = re.search(r"\[[^0-9]+,* *[^0-9]*\]", clue)
    if nonverbal:
        clue = re.sub(r"\[|\]", "", clue)
        clue = clue + " (nonverbal)"

    if clue[:4] == '""" ' and clue[-4:] == ' """':
        clue = '"' + clue[4:-4] + '"'
    if clue[:4] == "''' " and clue[-4:] == " '''":
        clue = "'" + clue[4:-4] + "'"
    if clue[:3] == '"""' and clue[-3:] == '"""':
        clue = '"' + clue[3:-3] + '"'
    if clue[:3] == "'''" and clue[-3:] == "'''":
        clue = "'" + clue[3:-3] + "'"

    return clue


# ======================================================================
# Grid printing  (matches Berkeley's Utils.print_grid)
# ======================================================================
def print_grid(letter_grid: List[List[str]]) -> None:
    """Pretty-print a crossword grid to stdout."""
    for row in letter_grid:
        row_display = [" " if val == "" else val for val in row]
        print("".join(row_display), flush=True)


# ======================================================================
# Answer-set & word-flip utilities  (matches Berkeley's Utils)
# ======================================================================
_WORDS_ALPHA: Optional[set] = None


def _load_words_alpha(path: str = "solver/words_alpha.txt") -> set:
    global _WORDS_ALPHA
    if _WORDS_ALPHA is None:
        try:
            _WORDS_ALPHA = {
                line.strip() for line in open(path, "r") if line.strip()
            }
        except FileNotFoundError:
            _WORDS_ALPHA = set()
    return _WORDS_ALPHA


def get_word_flips(fill: str, num_candidates: int = 10) -> List[str]:
    """Try flipping each letter and return fills that segment into the fewest real words."""
    try:
        import wordsegment
        from wordsegment import segment, clean as ws_clean

        wordsegment.load()
    except ImportError:
        return [fill.upper()]

    dictionary = _load_words_alpha()

    def _num_words(text: str) -> Tuple[int, float]:
        segmented = segment(text)
        prob = 0.0
        for word in segmented:
            if word not in dictionary:
                return 999, -9999999999999
            prob += math.log(max(wordsegment.UNIGRAMS.get(word, 1), 1))
        return len(segmented), prob

    fill = ws_clean(fill)
    results: Dict[int, list] = {}
    min_len = 999
    for idx, char in enumerate(fill):
        for new_letter in string.ascii_lowercase:
            new_fill = list(fill)
            new_fill[idx] = new_letter
            new_fill = "".join(new_fill)
            nw, prob = _num_words(new_fill)
            results.setdefault(nw, []).append((new_fill, prob))
            if nw < min_len:
                min_len = nw

    if min_len == 999:
        return [fill.upper()]

    all_results = sum(
        [sorted(results[k], key=lambda x: -x[1]) for k in sorted(results.keys())],
        [],
    )
    return [a[0].upper() for a in all_results[:num_candidates]]
