import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import time
import pandas as pd


@dataclass
class EloConfig:
    initial_rating: float = 1500.0
    k_factor: float = 32.0
    scale: float = 400.0  # Elo scale constant


def expected_score(r_a: float, r_b: float, scale: float) -> float:
    """Expected score for A vs B under Elo."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / scale))

def get_scores(games_won_play, games_won_draw) -> Tuple[float, float]:
    """Determine scores for player A and B based on games won."""
    if games_won_play > games_won_draw:
        if games_won_draw == 1:
            return 0.75, 0.25  # Player A wins with a draw
        return 1.0, 0.0  # Player A wins
    elif games_won_play < games_won_draw:
        if games_won_play == 1:
            return 0.25, 0.75  # Player B wins with a draw
        return 0.0, 1.0  # Player B wins
    else:
        return 0.5, 0.5  # Draw

def compute_elo_from_csv(
    csv_path: str,
    player_a_col: str,
    player_b_col: str,
    date_col: Optional[str] = None,
    config: EloConfig = EloConfig(),
    history_out_path: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {player_a_col, player_b_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

    # If a date column is provided, process chronologically
    if date_col:
        if date_col not in df.columns:
            raise KeyError(f"date_col={date_col!r} not found in CSV columns.")
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        # Put unknown dates last but keep stable ordering otherwise
        df = df.sort_values(by=[date_col], na_position="last").reset_index(drop=True)

    ratings: Dict[str, float] = {}

    def get_rating(p: str) -> float:
        if p not in ratings:
            ratings[p] = float(config.initial_rating)
        return ratings[p]

    history_rows = []

    for i, row in df.iterrows():
        a = str(row[player_a_col]).strip()
        b = str(row[player_b_col]).strip()
        if not a or a.lower() == "nan" or not b or b.lower() == "nan":
            raise ValueError(f"Row {i}: player names missing/invalid: A={a!r}, B={b!r}")
        score_a, score_b = get_scores(row['Games Won Play'], row['Games Won Draw'])

        r_a = get_rating(a)
        r_b = get_rating(b)

        exp_a = expected_score(r_a, r_b, config.scale)
        exp_b = 1.0 - exp_a

        new_r_a = r_a + config.k_factor * (score_a - exp_a)
        new_r_b = r_b + config.k_factor * (score_b - exp_b)

        ratings[a] = new_r_a
        ratings[b] = new_r_b

        history_rows.append(
            {
                "match_index": i,
                "player_a": a,
                "player_b": b,
                "outcome_raw": 'Play' if score_a > score_b else 'Draw' if score_a < score_b else 'Tie',
                "score_a": score_a,
                "score_b": score_b,
                "rating_a_before": r_a,
                "rating_b_before": r_b,
                "rating_a_after": new_r_a,
                "rating_b_after": new_r_b,
                "expected_a": exp_a,
                "expected_b": exp_b,
                **({date_col: row[date_col]} if date_col else {}),
            }
        )

    final = (
        pd.DataFrame([{"Deck": p, "mmr": r} for p, r in ratings.items()])
        .sort_values("mmr", ascending=False)
        .reset_index(drop=True)
    )
    current_time = time.strftime("%Y-%m-%d", time.localtime())
    final.to_csv(f"final_ratings_{current_time}.csv", index=False)
    if history_out_path:
        pd.DataFrame(history_rows).to_csv(history_out_path, index=False)

    return final


def main():
    parser = argparse.ArgumentParser(description="Compute Elo/MMR from a match-results CSV.")
    parser.add_argument("csv", help="Path to input CSV")
    parser.add_argument("--player-a-col", default="Deck on Play", help="Column name for player/team A")
    parser.add_argument("--player-b-col", default="Deck on Draw", help="Column name for player/team B")
    parser.add_argument("--date-col", default=None, help="Optional column name for match date/time")
    parser.add_argument("--initial", type=float, default=1500.0, help="Initial rating for new players")
    parser.add_argument("--k", type=float, default=32.0, help="K-factor (rating volatility)")
    parser.add_argument("--scale", type=float, default=400.0, help="Elo scale constant")
    parser.add_argument("--history-out", default=None, help="Optional output CSV for per-match history")
    args = parser.parse_args()

    config = EloConfig(initial_rating=args.initial, k_factor=args.k, scale=args.scale)

    final = compute_elo_from_csv(
        csv_path=args.csv,
        player_a_col=args.player_a_col,
        player_b_col=args.player_b_col,
        date_col=args.date_col,
        config=config,
        history_out_path=args.history_out,
    )

    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
