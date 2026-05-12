import csv
from collections import defaultdict
from pathlib import Path

RESULTS_FILE = Path("game_results.csv")


def summarize_results():
    """Print a compact comparison of human and AI runs from game_results.csv.

    The environment appends one row per episode. This script groups those rows by
    run label and level, then reports completion rate plus timing/input stats.
    """
    if not RESULTS_FILE.exists():
        print("No game_results.csv file found yet.")
        return

    rows = _read_rows()
    if not rows:
        print("game_results.csv does not contain any readable result rows yet.")
        return

    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[(row["run_label"], row["level"], row["generation"])].append(row)

    for (run_label, level, generation), group in sorted(grouped_rows.items()):
        completed = [row for row in group if row["outcome"] == "complete"]
        completion_rate = len(completed) / len(group) * 100

        generation_label = f"generation {generation}" if generation else "no generation"
        print(f"{run_label} - {level} - {generation_label}")
        print(f"  episodes: {len(group)}")
        print(f"  completed: {len(completed)} ({completion_rate:.1f}%)")

        if completed:
            print(f"  best ticks: {_min_number(completed, 'ticks'):.0f}")
            print(f"  avg ticks: {_average(completed, 'ticks'):.1f}")
            print(f"  best inputs: {_min_number(completed, 'inputs'):.0f}")
            print(f"  avg inputs: {_average(completed, 'inputs'):.1f}")
            print(f"  best time: {_min_number(completed, 'elapsed_seconds'):.3f}s")
            print(f"  avg time: {_average(completed, 'elapsed_seconds'):.3f}s")
        print()


def _read_rows():
    rows = []
    with RESULTS_FILE.open("r", encoding="utf-8", newline="") as results_file:
        reader = csv.DictReader(results_file, delimiter=";")
        for row in reader:
            normalized = _normalize_row(row)
            if normalized is not None:
                rows.append(normalized)
    return rows


def _normalize_row(row):
    required = ["run_label", "level", "ticks", "inputs", "elapsed_seconds"]
    if not all(row.get(field) for field in required):
        return None

    return {
        "run_label": row["run_label"],
        "level": row["level"],
        "generation": row.get("generation") or "",
        "episode": row.get("episode") or "",
        "outcome": row.get("outcome") or "complete",
        "ticks": float(row["ticks"]),
        "inputs": float(row["inputs"]),
        "elapsed_seconds": float(row["elapsed_seconds"]),
    }


def _average(rows, field):
    return sum(row[field] for row in rows) / len(rows)


def _min_number(rows, field):
    return min(row[field] for row in rows)


if __name__ == "__main__":
    summarize_results()
