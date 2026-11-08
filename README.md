# Poker-Texas-Hold-em-Assistant
Interactive Texas Hold’em tool using python that estimates an opponent’s hand range, including bluff probability from bet sizing, and estimates win probability via Monte Carlo simulation with confidence intervals; recommends the user fold/call/raise actions from pot odds and expected value.

Enter your hand and whatever shared cards are already on the table. The program:

  1. Guesses which hands your opponent could hold, INCLUDING BLUFFS,
     based on whether they checked or bet (and how much).
  2. Plays out thousands of random games (Monte Carlo simulation) to estimate
     your win probability against both their real hands and their bluffs.
  3. Compares that with the pot odds and recommends FOLD / CALL / RAISE
     (or CHECK / BET if they checked).
  4. Asks what happened next (your action, then the next card, then their
     action) and repeats until the hand ends. It never asks for a card you
     have already entered.

Run:  python3 Poker.py
Cards: rank + suit, e.g. Ah = Ace of hearts, 10c = 10 of clubs, Ks = King of spades.
"""
import math
import random
from collections import Counter
from itertools import combinations

# ----------------------------------------------------------------------
# 1. CARDS AND HAND EVALUATION
# ----------------------------------------------------------------------
RANKS = "23456789TJQKA"          # T is used internally for 10
SUITS = "shdc"
DECK = [r + s for r in RANKS for s in SUITS]
RANK_VALUE = {r: i for i, r in enumerate(RANKS, start=2)}
HAND_NAMES = ["High card", "Pair", "Two pair", "Three of a kind", "Straight",
              "Flush", "Full house", "Four of a kind", "Straight flush"]


def score_five(cards):
    """Score a 5-card hand as (category, tiebreakers). Higher is better."""
    values = sorted((RANK_VALUE[c[0]] for c in cards), reverse=True)
    suits = [c[1] for c in cards]
    counts = Counter(values)
    ordered = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    groups = [c for _, c in ordered]
    tiebreak = [v for v, _ in ordered]

    flush = len(set(suits)) == 1
    straight = len(counts) == 5 and values[0] - values[4] == 4
    if set(values) == {14, 2, 3, 4, 5}:        # the wheel: A-2-3-4-5
        straight, tiebreak = True, [5, 4, 3, 2, 1]

    if straight and flush:      return (8, tiebreak)
    if groups == [4, 1]:        return (7, tiebreak)
    if groups == [3, 2]:        return (6, tiebreak)
    if flush:                   return (5, tiebreak)
    if straight:                return (4, tiebreak)
    if groups == [3, 1, 1]:     return (3, tiebreak)
    if groups == [2, 2, 1]:     return (2, tiebreak)
    if groups == [2, 1, 1, 1]:  return (1, tiebreak)
    return (0, tiebreak)


def best_hand(cards):
    """Best 5-card score from 5, 6 or 7 cards."""
    if len(cards) == 5:
        return score_five(cards)
    return max(score_five(combo) for combo in combinations(cards, 5))


# ----------------------------------------------------------------------
# 2. OPPONENT RANGE MODEL (real hands + bluffs)
# ----------------------------------------------------------------------
STYLE_MULT = {"t": 0.5, "n": 1.0, "a": 1.6}   # tight / normal / aggressive


def preflop_strength(c1, c2):
    """Rough rule-of-thumb strength of a 2-card starting hand."""
    v1, v2 = sorted((RANK_VALUE[c1[0]], RANK_VALUE[c2[0]]), reverse=True)
    s = v1 * 2 + v2
    if v1 == v2:
        s += 30 + v1 * 2           # pairs are strong
    if c1[1] == c2[1]:
        s += 6                     # suited
    gap = v1 - v2
    if gap == 1:
        s += 4                     # connected
    elif gap == 2:
        s += 2
    elif gap >= 4 and v1 != v2:
        s -= 3
    return (s,)


def opponent_range(hero, board, action, bet, pot, style="n"):
    """
    Return (value_hands, bluff_hands, bluff_prob).

    Every possible opponent hand is ranked strongest -> weakest, then:
      check: they probably don't have a monster -> drop the top 15%. No bluffs
             (a check isn't a bet), so bluff_prob = 0.
      bet:   VALUE hands = a slice of the strongest hands (bigger bet -> tighter).
             BLUFF hands = drawn from the weakest 40% of hands.
             bluff_prob = bet / (pot + 2*bet), the bluff share of a balanced
             betting range, scaled by their style (tight x0.5, aggressive x1.6).
    These numbers are ASSUMPTIONS; real opponents vary a lot.
    """
    known = set(hero) | set(board)
    remaining = [c for c in DECK if c not in known]
    combos = list(combinations(remaining, 2))

    def strength(combo):
        if len(board) >= 3:
            return best_hand(list(combo) + list(board))
        return preflop_strength(*combo)

    combos.sort(key=strength, reverse=True)
    n = len(combos)

    if action == "any":
        return combos, [], 0.0
    if action == "check":
        return combos[int(0.15 * n):], [], 0.0

    ratio = bet / pot if pot > 0 else 1.0
    keep = 0.60 if ratio <= 0.33 else 0.45 if ratio <= 0.75 else 0.35 if ratio <= 1.25 else 0.25
    value = combos[:max(1, int(keep * n))]
    bluffs = combos[int(0.6 * n):]
    q = min(0.5, bet / (pot + 2 * bet) * STYLE_MULT[style])
    return value, bluffs, q


def range_breakdown(value, bluffs, q, board):
    """Share of the opponent's range in each hand category (needs a flop)."""
    if len(board) < 3:
        return None
    share = Counter()
    for c in value:
        share[best_hand(list(c) + list(board))[0]] += (1 - q) / len(value)
    for c in bluffs:
        share[best_hand(list(c) + list(board))[0]] += q / len(bluffs)
    return [(HAND_NAMES[k], share[k]) for k in sorted(share, reverse=True)]


# ----------------------------------------------------------------------
# 3. MONTE CARLO SIMULATION
# ----------------------------------------------------------------------
def simulate(hero, board, opp_hands, n_sims):
    """Return (win, tie, loss) fractions against hands drawn from opp_hands."""
    wins = ties = losses = 0
    need = 5 - len(board)
    base_used = set(hero) | set(board)
    for _ in range(n_sims):
        opp = random.choice(opp_hands)
        used = base_used | set(opp)
        available = [c for c in DECK if c not in used]
        full_board = list(board) + random.sample(available, need)
        h = best_hand(list(hero) + full_board)
        o = best_hand(list(opp) + full_board)
        if h > o:
            wins += 1
        elif h == o:
            ties += 1
        else:
            losses += 1
    return wins / n_sims, ties / n_sims, losses / n_sims


# ----------------------------------------------------------------------
# 4. DECISION LOGIC
# ----------------------------------------------------------------------
def recommend(eq, M, c, opp_in, q):
    """
    Return (action, explanation).
      M = chips you would win right now (everything in the middle)
      c = chips you must add to call (0 if they checked)
      q = chance their bet is a bluff
    """
    if c > 0:
        need = c / (M + c)                 # break-even equity (pot odds)
        ev = eq * M - (1 - eq) * c         # EV of calling
        bluff_note = f" This already counts a {q:.0%} chance they're bluffing." if q > 0 else ""
        if eq >= 0.65 and eq >= need + 0.25:
            raise_to = opp_in + M + c      # call, then raise by the pot
            return ("RAISE",
                    f"Equity {eq:.0%} is far above the {need:.0%} needed to call. "
                    f"A pot-sized raise (to about {raise_to:.0f} total this round) "
                    f"builds the pot while you're ahead.{bluff_note}")
        if ev > 0:
            return ("CALL",
                    f"Equity {eq:.0%} beats the {need:.0%} break-even, so calling "
                    f"has positive EV ({ev:+.1f} chips).{bluff_note}")
        return ("FOLD",
                f"Equity {eq:.0%} is below the {need:.0%} break-even, so calling "
                f"loses {abs(ev):.1f} chips on average.{bluff_note}")
    if eq >= 0.65:
        return ("BET",
                f"You're a clear favourite ({eq:.0%}). Bet about {0.66 * M:.0f} "
                f"(2/3 pot) to get value from worse hands.")
    return ("CHECK",
            f"Equity {eq:.0%} isn't strong enough to bet for value. "
            f"Checking keeps the pot small and lets you see the next card for free.")


# ----------------------------------------------------------------------
# 5. USER INPUT HELPERS
# ----------------------------------------------------------------------
class Quit(Exception):
    pass


def pretty(cards):
    """Show cards the way the user types them (10 instead of T)."""
    return " ".join(c.replace("T", "10") for c in cards)


def normalise(token):
    token = token.strip().replace("10", "T")
    if len(token) != 2:
        return None
    card = token[0].upper() + token[1].lower()
    return card if card in DECK else None


def ask_cards(prompt, count, used, optional=False):
    """Ask until the user enters exactly `count` valid, unused cards."""
    while True:
        raw = input(prompt).replace(",", " ").split()
        if raw and raw[0].lower() == "q":
            raise Quit
        if optional and not raw:
            return []
        cards = [normalise(t) for t in raw]
        if len(cards) != count or None in cards:
            print(f"  -> Enter exactly {count} card(s), e.g. Ah 10c (rank then suit)")
            continue
        if len(set(cards)) != count or any(c in used for c in cards):
            print("  -> That includes a duplicate or a card already used.")
            continue
        return cards


def ask_number(prompt, minimum=0.0, default=None):
    """Ask for a number. If `default` is given, pressing Enter accepts it."""
    while True:
        raw = input(prompt).strip()
        if raw.lower() == "q":
            raise Quit
        if raw == "" and default is not None:
            return default
        try:
            value = float(raw)
            if value >= minimum:
                return value
        except ValueError:
            pass
        print(f"  -> Please enter a number >= {minimum:g}")


def ask_choice(prompt, options):
    """Ask until the user types one of the allowed letters."""
    while True:
        a = input(prompt).strip().lower()
        if a == "q":
            raise Quit
        if a in options:
            return a
        print(f"  -> Type one of: {', '.join(options)}")


# ----------------------------------------------------------------------
# 6. ANALYSIS OF ONE DECISION
# ----------------------------------------------------------------------
def analyse(hero, board, pot0, hero_in, opp_in, n_sims, style):
    """Print range, bluff check, win chances, risk and a recommendation."""
    M = pot0 + hero_in + opp_in            # chips in the middle right now
    c = opp_in - hero_in                   # chips you must add to call
    action = "check" if opp_in == 0 else "bet"

    print("\nRunning simulated games... (a few seconds)")
    value, bluffs, q = opponent_range(hero, board, action, c, M - c, style)
    wv, tv, lv = simulate(hero, board, value, n_sims)
    eq_value = wv + 0.5 * tv
    if bluffs and q > 0:
        wb, tb, lb = simulate(hero, board, bluffs, n_sims)
        eq_bluff = wb + 0.5 * tb
    else:
        wb = tb = lb = eq_bluff = 0.0
    # blend: with probability q they are bluffing, otherwise they have a real hand
    w = (1 - q) * wv + q * wb
    t = (1 - q) * tv + q * tb
    l = (1 - q) * lv + q * lb
    eq = w + 0.5 * t
    se = math.sqrt(eq * (1 - eq) / (n_sims * 2))
    everyone, _, _ = opponent_range(hero, board, "any", 0, 1)
    wr, tr, _ = simulate(hero, board, everyone, n_sims)
    eq_random = wr + 0.5 * tr

    print("\n" + "=" * 52)
    print(f"Your hand: {pretty(hero)}    Board: {pretty(board) if board else '(none yet)'}")
    print("=" * 52)

    what = "checked" if action == "check" else f"bet/raised to {opp_in:.0f}"
    print(f"\nWHAT THEY MIGHT HOLD  (opponent {what})")
    print(f"  Hands considered possible: {len(value) + len(bluffs)}")
    breakdown = range_breakdown(value, bluffs, q, board)
    if breakdown:
        for name, share in breakdown:
            if share >= 0.01:
                print(f"  {name:<16}{share:6.1%}")
    else:
        print("  (Before the flop hands are ranked by a simple rule of thumb;")
        print("   no breakdown until the flop is dealt.)")
    print("  Note: a range of possibilities, not a prediction of their exact cards.")

    print("\nBLUFF CHECK")
    if action == "bet":
        print(f"  Chance this bet is a BLUFF (assumed): {q:.0%}")
        print(f"  If they have a real hand ({1 - q:.0%}): you win {eq_value:.0%} of the time")
        print(f"  If they are bluffing     ({q:.0%}): you win {eq_bluff:.0%} of the time")
    else:
        print("  A check can't be a bluff (they didn't bet), but they could be")
        print("  trapping - checking a strong hand. The model allows for this by")
        print("  only removing their very top hands, not all strong ones.")

    print("\nYOUR CHANCES")
    print(f"  Win {w:.1%}   Tie {t:.1%}   Lose {l:.1%}")
    print(f"  Overall equity: {eq:.1%}  (margin of error about +/-{1.96 * se:.1%})")
    print(f"  For comparison, equity vs a totally random hand: {eq_random:.1%}")

    print("\nRISK")
    if c > 0:
        need = c / (M + c)
        print(f"  Pot is {M:.0f}. To call you risk {c:.0f} to win {M:.0f}.")
        print(f"  Break-even equity (pot odds): {need:.1%}")
        print(f"  EV of calling: {eq * M - (1 - eq) * c:+.1f} chips")
    else:
        print(f"  Pot is {M:.0f}. Checking is free; betting risks chips to win more.")

    decision, why = recommend(eq, M, c, opp_in, q)
    print("\n" + "-" * 52)
    print(f"RECOMMENDATION: {decision}")
    print(f"  {why}")
    print("-" * 52)


# ----------------------------------------------------------------------
# 7. PLAY ONE HAND
# ----------------------------------------------------------------------
STREETS = ["PREFLOP", "FLOP", "TURN", "RIVER"]
BOARD_SIZE = {0: 0, 1: 3, 2: 4, 3: 5}          # shared cards visible per round
CARD_LABEL = {1: "the 3 flop cards", 2: "the turn card", 3: "the river card"}


def play_hand(n_sims=4000):
    used = set()
    hero = ask_cards("Your two cards (e.g. Ah 10c): ", 2, used)
    used |= set(hero)
    style = ask_choice("How often does this opponent bluff? "
                       "[t]ight/rarely  [n]ormal  [a]ggressive/often: ", ("t", "n", "a"))

    # --- shared cards already on the table (press Enter if not dealt yet) ---
    print("\nShared cards on the table so far (press Enter if not dealt yet):")
    board = []
    flop = ask_cards("  Flop (3 cards): ", 3, used, optional=True)
    if flop:
        board += flop
        used |= set(flop)
        turn = ask_cards("  Turn (1 card): ", 1, used, optional=True)
        if turn:
            board += turn
            used |= set(turn)
            river = ask_cards("  River (1 card): ", 1, used, optional=True)
            if river:
                board += river
                used |= set(river)

    street = {0: 0, 3: 1, 4: 2, 5: 3}[len(board)]   # which round we are in now
    pot = None

    while street <= 3:
        print("\n" + "#" * 52)
        print(f"#  {STREETS[street]}")
        print("#" * 52)

        # --- ask ONLY for shared cards we don't have yet ---
        missing = BOARD_SIZE[street] - len(board)
        if missing > 0:
            new = ask_cards(f"What happened next? Enter {CARD_LABEL[street]}: ", missing, used)
            board += new
            used |= set(new)
        if board:
            print(f"Board: {pretty(board)}")

        # --- pot at the start of this round ---
        if pot is None:
            pot0 = ask_number("Chips in the pot before their action (include blinds): ", minimum=1)
        else:
            pot0 = ask_number(f"Pot is {pot:.0f}. Press Enter to accept, or type the real pot: ",
                              minimum=1, default=pot)

        # --- opponent acts first on every round (assumption) ---
        hero_in = 0.0
        opp_in = 0.0
        a = ask_choice("What did your opponent do? [c]heck or [b]et/raise: ", ("c", "b"))
        if a == "b":
            opp_in = ask_number("  How many chips did they bet? ", minimum=1)

        # --- decision loop for this round (handles raises and re-raises) ---
        while True:
            analyse(hero, board, pot0, hero_in, opp_in, n_sims, style)
            opts = "[f]old, [c]all" if opp_in > hero_in else "[f]old, [c]heck"
            mine = ask_choice(f"\nWhat did you do? {opts}, [r]aise/bet: ", ("f", "c", "r"))

            if mine == "f":
                print("\nYou folded. Hand over.")
                return
            if mine == "c":
                hero_in = opp_in
                break
            hero_in = ask_number("  Raise/bet to how many chips in total this round? ",
                                 minimum=opp_in + 1)
            reply = ask_choice("  What did the opponent do? [c]all, [f]old, [r]e-raise: ",
                               ("c", "f", "r"))
            if reply == "f":
                print(f"\nThey folded. You win the pot of {pot0 + hero_in + opp_in:.0f} chips!")
                return
            if reply == "c":
                opp_in = hero_in
                break
            opp_in = ask_number("  They re-raised to how many chips in total? ",
                                minimum=hero_in + 1)

        pot = pot0 + hero_in + opp_in
        print(f"\nRound complete. Pot is now {pot:.0f} chips.")
        street += 1

    print("\n" + "#" * 52)
    print("#  SHOWDOWN - all cards are out. Compare hands to see who wins!")
    print("#" * 52)


# ----------------------------------------------------------------------
# 8. MAIN
# ----------------------------------------------------------------------
def main():
    print("\n=== POKER DECISION ASSISTANT v4 (heads-up Texas Hold'em) ===")
    print("HOW TO TYPE CARDS: rank + suit, with a space between cards.")
    print("  Ranks: 2-10, J=Jack, Q=Queen, K=King, A=Ace")
    print("  Suits: s=spades, h=hearts, d=diamonds, c=clubs")
    print("  Examples: Ah = Ace of hearts | Ks = King of spades | 7c = 7 of clubs | 10c = 10 of clubs")
    print("\nHOW IT WORKS: enter your cards and whatever is on the table. It guesses")
    print("what your opponent could hold (including bluffs), plays out thousands of")
    print("random games to estimate your win chance, compares that with the bet, and")
    print("advises you. Then it asks what happened next and repeats until the hand ends.")
    print("Assumes your opponent acts first each round. Type q at any prompt to quit.\n")

    try:
        while True:
            play_hand()
            again = ask_choice("\nPlay another hand? [y]es / [n]o: ", ("y", "n"))
            if again == "n":
                break
            print("\n" + "=" * 52 + "\nNEW HAND\n" + "=" * 52)
    except (Quit, KeyboardInterrupt, EOFError):
        pass
    print("\nLimits: the opponent model is a simple assumption, and draws")
    print("(flush/straight draws) are not given extra weight in their range.")
    print("Thanks for playing!\n")


if __name__ == "__main__":
    main()
