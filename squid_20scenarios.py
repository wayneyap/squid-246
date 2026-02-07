"""
Squid Game — 20 Realistic Scenarios (v3 - Early Termination)
$5k/squid | 9 players | 12 squids | Win rate = 1/9 per hand

Rules:
  - 11 hands: Hand 1 = 2 squids, Hands 2-11 = 1 squid each (12 total)
  - Game ends when exactly 1 player has 0 squids (early termination)
  - OR after all hands are played (whichever comes first)
  - Winners (>0 squids): earn payout(squids) × (# players at 0)
  - Losers (0 squids): pay Σ payout(opp_j) for all opponents
  - Zero-sum game

Payout ($5k base):
  1=$5k 2=$10k 3=$30k 4=$40k 5=$100k 6=$120k 7=$210k 8=$240k 9=$360k
"""

import numpy as np

BASE = 5000
PAYOUTS = {0:0, 1:5000, 2:10000, 3:30000, 4:40000,
           5:100000, 6:120000, 7:210000, 8:240000, 9:360000}
TOTAL_SQUIDS = 12
NUM_PLAYERS = 9
MAX_SQ = 9
N_SIMS = 800_000

def payout(n):
    return PAYOUTS.get(min(n, MAX_SQ), PAYOUTS[MAX_SQ])

def fmt(n):
    if n < 0: return f"-${abs(n):,.0f}"
    return f"${n:,.0f}"

PAY_LOOKUP = np.array([payout(i) for i in range(25)])


def compute_net(my_sq, opp_sq):
    """Net earning/loss at game end."""
    my_pay = PAY_LOOKUP[np.minimum(my_sq, MAX_SQ)]
    opp_zeros = (opp_sq == 0).sum(axis=1)
    winner_earn = my_pay * opp_zeros
    opp_pays = PAY_LOOKUP[np.minimum(opp_sq, MAX_SQ)]
    loser_penalty = -opp_pays.sum(axis=1)
    return np.where(my_sq > 0, winner_earn, loser_penalty)


def check_game_end(my_sq, opp_sq):
    """True where exactly 1 player total has 0 squids."""
    my_zero = (my_sq == 0).astype(int)
    opp_zeros = (opp_sq == 0).sum(axis=1)
    return (my_zero + opp_zeros) == 1


def simulate_hand_value(my_sq, opp_sq, remaining, n_sims=N_SIMS):
    """Monte Carlo: value of winning the next hand (with early termination)."""
    opp = np.array(opp_sq, dtype=int)

    # WIN PATH: I get +1 squid
    my_w = np.full(n_sims, my_sq + 1, dtype=int)
    opp_w = np.tile(opp, (n_sims, 1))
    ended_w = check_game_end(my_w, opp_w).copy()

    # LOSE PATH: random opponent gets +1 squid
    my_l = np.full(n_sims, my_sq, dtype=int)
    opp_l = np.tile(opp, (n_sims, 1))
    rng_r = np.random.default_rng(99)
    recipient = rng_r.integers(0, 8, size=n_sims)
    opp_l[np.arange(n_sims), recipient] += 1
    ended_l = check_game_end(my_l, opp_l).copy()

    # Future hands (same RNG for both paths = variance reduction)
    rng = np.random.default_rng(42)
    for _ in range(remaining - 1):
        winner = rng.integers(0, 9, size=n_sims)
        active_w = ~ended_w
        active_l = ~ended_l

        i_win = (winner == 0)
        my_w += (i_win & active_w).astype(int)
        my_l += (i_win & active_l).astype(int)

        for k in range(8):
            opp_win = (winner == k + 1)
            opp_w[:, k] += (opp_win & active_w).astype(int)
            opp_l[:, k] += (opp_win & active_l).astype(int)

        ended_w |= active_w & check_game_end(my_w, opp_w)
        ended_l |= active_l & check_game_end(my_l, opp_l)

    earn_w = compute_net(my_w, opp_w)
    earn_l = compute_net(my_l, opp_l)

    # Penalty stats for lose path
    zero_l = (my_l == 0)
    p_zero_l = zero_l.mean()
    avg_pen_l = 0.0
    if zero_l.any():
        avg_pen_l = PAY_LOOKUP[np.minimum(opp_l[zero_l], MAX_SQ)].sum(axis=1).mean()

    # Early termination stats
    p_early_w = ended_w.mean()
    p_early_l = ended_l.mean()

    return {
        'value': (earn_w - earn_l).mean(),
        'ev_win': earn_w.mean(),
        'ev_lose': earn_l.mean(),
        'p_zero_lose': p_zero_l,
        'avg_penalty_lose': avg_pen_l,
        'p_early_w': p_early_w,
        'p_early_l': p_early_l,
    }


def simulate_baseline(my_sq, opp_sq, remaining, n_sims=N_SIMS):
    """Expected net if you play out remaining hands at 1/9 (with early termination)."""
    opp = np.array(opp_sq, dtype=int)
    my_f = np.full(n_sims, my_sq, dtype=int)
    opp_f = np.tile(opp, (n_sims, 1))
    ended = check_game_end(my_f, opp_f).copy()

    rng = np.random.default_rng(42)
    for _ in range(remaining):
        active = ~ended
        winner = rng.integers(0, 9, size=n_sims)
        my_f += ((winner == 0) & active).astype(int)
        for k in range(8):
            opp_f[:, k] += ((winner == k + 1) & active).astype(int)
        ended |= active & check_game_end(my_f, opp_f)

    earn = compute_net(my_f, opp_f)

    zero_f = (my_f == 0)
    p_zero = zero_f.mean()
    avg_pen = 0.0
    if zero_f.any():
        avg_pen = PAY_LOOKUP[np.minimum(opp_f[zero_f], MAX_SQ)].sum(axis=1).mean()

    # Count losers
    n_losers = (my_f == 0).astype(int) + (opp_f == 0).sum(axis=1)
    avg_losers = n_losers.mean()

    return {'ev': earn.mean(), 'p_zero': p_zero, 'avg_penalty': avg_pen, 'avg_losers': avg_losers}


scenarios = [
    # --- EARLY GAME ---
    {'id': 1, 'phase': 'EARLY', 'out': 5, 'remain': 7,
     'title': "Shut out early — 0 squids, one opp ran hot",
     'you': 0, 'opp': [3,1,1,0,0,0,0,0],
     'note': "One opp won opener+1. You have nothing. 5 opps at 0."},
    {'id': 2, 'phase': 'EARLY', 'out': 5, 'remain': 7,
     'title': "Won the opener — sitting at 2 squids",
     'you': 2, 'opp': [1,1,1,0,0,0,0,0],
     'note': "You won double-hand (2 sq). Three others have 1 each."},
    {'id': 3, 'phase': 'EARLY', 'out': 5, 'remain': 7,
     'title': "Got 1 squid — average position",
     'you': 1, 'opp': [2,1,1,0,0,0,0,0],
     'note': "You picked up 1. One opp has 2 (won opener). Need more."},
    {'id': 4, 'phase': 'EARLY', 'out': 6, 'remain': 6,
     'title': "Blanked through 5 hands — squids spread",
     'you': 0, 'opp': [2,2,1,1,0,0,0,0],
     'note': "Everyone else getting squids. 4 opps still at 0."},
    {'id': 5, 'phase': 'EARLY', 'out': 6, 'remain': 6,
     'title': "Running hot — 3 squids at 2x tier",
     'you': 3, 'opp': [1,1,1,0,0,0,0,0],
     'note': "Table leader with 3 (2x). 5 opps at 0. Strong spot."},
    # --- MID GAME ---
    {'id': 6, 'phase': 'MID', 'out': 7, 'remain': 5,
     'title': "Shut out through mid-game — 0 squids",
     'you': 0, 'opp': [3,2,1,1,0,0,0,0],
     'note': "Deep in game with nothing. Max reachable = 5 squids."},
    {'id': 7, 'phase': 'MID', 'out': 7, 'remain': 5,
     'title': "Average mid-game — 2 squids, spread table",
     'you': 2, 'opp': [2,1,1,1,0,0,0,0],
     'note': "Typical spot. At 1x, need 1 more for 2x threshold."},
    {'id': 8, 'phase': 'MID', 'out': 7, 'remain': 5,
     'title': "2 squids but one opp dominant",
     'you': 2, 'opp': [3,1,1,0,0,0,0,0],
     'note': "One opp has 3 (2x). You're at 2. Fight for thresholds."},
    {'id': 9, 'phase': 'MID', 'out': 7, 'remain': 5,
     'title': "Strong — 4 squids, one from 4x",
     'you': 4, 'opp': [1,1,1,0,0,0,0,0],
     'note': "Dominating. One win = 4x ($100k payout). 5 opps at 0."},
    {'id': 10, 'phase': 'MID', 'out': 7, 'remain': 5,
     'title': "Decent — 3 squids at 2x, building",
     'you': 3, 'opp': [2,1,1,0,0,0,0,0],
     'note': "At 2x. Chasing 5 for the 4x jump. 5 opps at 0."},
    {'id': 11, 'phase': 'MID', 'out': 8, 'remain': 4,
     'title': "Behind mid-game — 2 squids, table filling",
     'you': 2, 'opp': [2,2,1,1,0,0,0,0],
     'note': "Only 4 hands left. At 1x. Need 3 for 2x minimum."},
    {'id': 12, 'phase': 'MID', 'out': 8, 'remain': 4,
     'title': "One from 4x — 4 squids, 4 hands left",
     'you': 4, 'opp': [2,1,1,0,0,0,0,0],
     'note': "Next win = 4x ($100k). 4 hands to chase 7 for 6x."},
    {'id': 13, 'phase': 'MID', 'out': 8, 'remain': 4,
     'title': "Crushing — 5 squids at 4x, chasing 6x",
     'you': 5, 'opp': [1,1,1,0,0,0,0,0],
     'note': "At 4x ($100k). Two more = 6x jackpot. 5 opps at 0."},
    # --- LATE GAME ---
    {'id': 14, 'phase': 'LATE', 'out': 9, 'remain': 3,
     'title': "Need 4x — 4 squids, 3 hands left",
     'you': 4, 'opp': [2,1,1,1,0,0,0,0],
     'note': "One win = 4x threshold. Three chances."},
    {'id': 15, 'phase': 'LATE', 'out': 9, 'remain': 3,
     'title': "Monster — 6 squids, chasing 6x jackpot",
     'you': 6, 'opp': [1,1,1,0,0,0,0,0],
     'note': "One win = 6x ($210k). Three chances at 1/9."},
    {'id': 16, 'phase': 'LATE', 'out': 9, 'remain': 3,
     'title': "Behind late — 2 squids, table nearly full",
     'you': 2, 'opp': [3,2,1,1,0,0,0,0],
     'note': "3 hands left. Max reach = 5 (4x). Likely stuck at 3."},
    {'id': 17, 'phase': 'LATE', 'out': 10, 'remain': 2,
     'title': "Need 4x — 4 squids, 2 hands left",
     'you': 4, 'opp': [2,2,1,1,0,0,0,0],
     'note': "Two shots at 4x. Win one = $40k→$100k jump."},
    {'id': 18, 'phase': 'LATE', 'out': 10, 'remain': 2,
     'title': "One from 6x — 6 squids, 2 hands left",
     'you': 6, 'opp': [2,1,1,0,0,0,0,0],
     'note': "Two shots at 6x. Win = $210k with few opp squids."},
    # --- ENDGAME ---
    {'id': 19, 'phase': 'ENDGAME', 'out': 11, 'remain': 1,
     'title': "LAST HAND — 6 squids, do-or-die for 6x",
     'you': 6, 'opp': [2,1,1,1,0,0,0,0],
     'note': "Win = 7sq (6x jackpot). Lose = stuck at 6 (4x)."},
    {'id': 20, 'phase': 'ENDGAME', 'out': 11, 'remain': 1,
     'title': "LAST HAND — 4 squids, do-or-die for 4x",
     'you': 4, 'opp': [3,2,1,1,0,0,0,0],
     'note': "Win = 5sq (4x). Lose = stuck at 4 (2x). Big jump."},
]


def main():
    print()
    print("=" * 78)
    print("  SQUID GAME — 20 REALISTIC SCENARIOS (v3 — Early Termination)")
    print("  $5k/squid | 9 players | 12 squids | Win rate = 1/9")
    print(f"  Monte Carlo: {N_SIMS:,} sims per scenario (variance-reduced)")
    print("=" * 78)
    print("\n  Payout: 1=$5k 2=$10k 3=$30k 4=$40k 5=$100k 6=$120k 7=$210k 8=$240k 9=$360k")
    print("  Thresholds: 3sq→2x  5sq→4x  7sq→6x  9sq→8x")

    print(f"\n{'─'*78}")
    print("  GAME RULES")
    print(f"{'─'*78}")
    print("  • Game ends when exactly 1 player has 0 squids (early termination)")
    print("  • OR after all 11 hands are played")
    print("  • Winners earn: payout(squids) × (# players at 0)")
    print("  • Losers (0 squids) pay: Σ of all opponents' payouts")
    print("  • Zero-sum: winners collect from losers")

    # ── Run all 20 scenarios ──
    results = []
    for sc in scenarios:
        you, opp, remain = sc['you'], sc['opp'], sc['remain']
        assert you + sum(opp) == sc['out']
        opp_zeros_now = sum(1 for x in opp if x == 0)
        total_at_0 = (1 if you == 0 else 0) + opp_zeros_now

        sim = simulate_hand_value(you, opp, remain)
        base = simulate_baseline(you, opp, remain)

        new_sq = you + 1
        threshold_hit = new_sq in [3, 5, 7, 9]
        threshold_label = {3: '2x', 5: '4x', 7: '6x', 9: '8x'}.get(new_sq, '')

        results.append({
            **sc, **sim,
            'baseline_ev': base['ev'],
            'p_zero_base': base['p_zero'],
            'avg_penalty_base': base['avg_penalty'],
            'avg_losers': base['avg_losers'],
            'opp_zeros_now': opp_zeros_now,
            'total_at_0': total_at_0,
            'threshold_hit': threshold_hit,
            'threshold_label': threshold_label,
        })

    # ── Print each scenario ──
    current_phase = None
    for r in results:
        if r['phase'] != current_phase:
            current_phase = r['phase']
            labels = {
                'EARLY': 'EARLY GAME (5-6 squids out, 6-7 remaining)',
                'MID':   'MID GAME (7-8 squids out, 4-5 remaining)',
                'LATE':  'LATE GAME (9-10 squids out, 2-3 remaining)',
                'ENDGAME': 'ENDGAME (11 squids out, 1 remaining)',
            }
            print(f"\n{'='*78}")
            print(f"  {labels[current_phase]}")
            print(f"{'='*78}")

        you = r['you']
        opp_str = ','.join(str(x) for x in sorted(r['opp'], reverse=True))

        print(f"\n  #{r['id']:<2} {r['title']}")
        print(f"  You: {you} sq | Opps: [{opp_str}] | {r['out']} out, {r['remain']} remain")
        print(f"  Players at 0: {r['total_at_0']} | {r['note']}")
        if r['threshold_hit']:
            print(f"  >> Next win triggers {r['threshold_label']} MULTIPLIER (→{you+1} squids)")

        print(f"  {'─'*60}")
        print(f"  Baseline EV:   {fmt(r['baseline_ev']):>12}  (avg {r['avg_losers']:.1f} losers)")
        if you == 0:
            print(f"    P(end at 0): {r['p_zero_base']*100:>5.1f}%  avg penalty: {fmt(r['avg_penalty_base'])}")
        print(f"  EV if WIN:     {fmt(r['ev_win']):>12}")
        print(f"  EV if LOSE:    {fmt(r['ev_lose']):>12}")
        if you == 0 and r['p_zero_lose'] > 0.001:
            print(f"    P(end at 0 if lose): {r['p_zero_lose']*100:.1f}%  avg penalty: {fmt(r['avg_penalty_lose'])}")
        print(f"  ★ HAND VALUE:  {fmt(r['value']):>12}")
        print(f"    P(early end): win={r['p_early_w']*100:.1f}% lose={r['p_early_l']*100:.1f}%")

    # ── SUMMARY TABLE ──
    print(f"\n{'='*78}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*78}\n")
    print(f"  {'#':<4} {'Phase':<8} {'You':<4} {'Out':<4} {'Left':<5} "
          f"{'@0':<4} {'Thr':<5} {'Value':>12}  {'EV(win)':>12}  {'EV(lose)':>12}  {'Baseline':>12}")
    print(f"  {'-'*95}")
    for r in results:
        thr = f"→{r['threshold_label']}" if r['threshold_hit'] else ""
        print(f"  {r['id']:<4} {r['phase']:<8} {r['you']:<4} {r['out']:<4} {r['remain']:<5} "
              f"{r['total_at_0']:<4} {thr:<5} {fmt(r['value']):>12}  "
              f"{fmt(r['ev_win']):>12}  {fmt(r['ev_lose']):>12}  {fmt(r['baseline_ev']):>12}")

    # ── ZERO-SQUID PENALTY DEEP DIVE ──
    print(f"\n{'='*78}")
    print(f"  ZERO-SQUID PENALTY DEEP DIVE")
    print(f"{'='*78}\n")
    zero_sc = [r for r in results if r['you'] == 0]
    print(f"  {'#':<4} {'Phase':<8} {'Remain':<7} {'P(end@0)':<10} "
          f"{'Avg Penalty':>12} {'Hand Value':>12} {'Baseline':>12}")
    print(f"  {'-'*70}")
    for r in zero_sc:
        print(f"  {r['id']:<4} {r['phase']:<8} {r['remain']:<7} "
              f"{r['p_zero_base']*100:>5.1f}%    {fmt(r['avg_penalty_base']):>12} "
              f"{fmt(r['value']):>12} {fmt(r['baseline_ev']):>12}")

    # ── HEURISTICS BY SQUID COUNT ──
    print(f"\n{'='*78}")
    print(f"  HEURISTICS BY SQUID COUNT")
    print(f"{'='*78}\n")
    by_count = {}
    for r in results:
        by_count.setdefault(r['you'], []).append(r)

    notes = {
        0: "HUGE — avoiding penalty worth $50-100k+",
        1: "Escaped zero, building toward 2x",
        2: "One from 2x — big payout jump ahead",
        3: "At 2x, building toward 4x",
        4: "One from 4x — $40k→$100k payout jump",
        5: "At 4x, chasing 6x jackpot",
        6: "One from 6x — THE JACKPOT ($120k→$210k)",
    }
    print(f"  {'Sq':<4} {'#':<4} {'Avg Value':>12}  {'Min':>12}  {'Max':>12}  Note")
    print(f"  {'-'*75}")
    for sq in sorted(by_count.keys()):
        vals = [r['value'] for r in by_count[sq]]
        avg = sum(vals) / len(vals)
        print(f"  {sq:<4} {len(vals):<4} {fmt(avg):>12}  {fmt(min(vals)):>12}  {fmt(max(vals)):>12}  {notes.get(sq,'')}")

    # ── GRAND MEAN ──
    all_vals = [r['value'] for r in results]
    grand = sum(all_vals) / len(all_vals)
    print(f"\n  Grand mean hand value (all 20): {fmt(grand)}")

    # ── JS REFERENCE DATA ──
    print(f"\n{'='*78}")
    print(f"  JAVASCRIPT REFERENCE DATA (copy to index.html)")
    print(f"{'='*78}\n")
    print("const SCENARIOS = [")
    for r in results:
        opp_str = ','.join(str(x) for x in sorted(r['opp'], reverse=True))
        print(f"  {{id:{r['id']:>2}, phase:'{r['phase']:<8s}', you:{r['you']}, "
              f"opp:'{opp_str}', left:{r['remain']}, "
              f"value:{r['value']:>8.0f}, baseline:{r['baseline_ev']:>8.0f}}},")
    print("];")
    print()


if __name__ == "__main__":
    main()
