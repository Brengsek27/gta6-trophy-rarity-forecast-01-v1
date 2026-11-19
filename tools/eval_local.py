import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = Path("/home/hatch/workspace/makefun/gta6-rarity-kit-fresh/gta6-trophy-rarity-forecast-01-kit")
STARTER = KIT / "starter/main.py"
CASES = KIT / "input"
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)


def logit(rate):
    rate = min(99.99, max(0.01, rate))
    return math.log(rate / (100 - rate))


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(program, case, work):
    input_dir, output_dir = work / "input", work / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / f"{case['id']}.in").write_text(json.dumps(case, separators=(",", ":")), encoding="utf-8")
    env = dict(os.environ, MAKEFUN_INPUT_DIR=str(input_dir), MAKEFUN_OUTPUT_DIR=str(output_dir))
    start = time.perf_counter()
    subprocess.run([sys.executable, str(program)], env=env, check=True, timeout=8, stdout=subprocess.DEVNULL)
    elapsed = time.perf_counter() - start
    output = output_dir / f"{case['id']}.out"
    raw = output.read_bytes()
    if len(raw) > 65536:
        raise AssertionError("output exceeds 65536 bytes")
    data = json.loads(raw)
    rows = data.get("trophies")
    expected = [t["id"] for t in case["trophies"]]
    if set(data) != {"trophies"} or not isinstance(rows, list) or len(rows) != len(expected):
        raise AssertionError("invalid top-level shape or row count")
    if [row.get("id") for row in rows] != expected:
        raise AssertionError("ids or row ordering do not match")
    for row in rows:
        if set(row) != {"id", "q"} or len(row["q"]) != 5:
            raise AssertionError("invalid row shape")
        q = row["q"]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100 for value in q):
            raise AssertionError("quantile outside finite 0..100")
        if q != sorted(q):
            raise AssertionError("quantiles are not non-decreasing")
    return raw, data, elapsed


def pinball(answer, forecasts):
    losses = []
    for y, row in zip(answer, forecasts):
        actual = logit(y)
        for tau, q in zip((0.05, 0.25, 0.5, 0.75, 0.95), row["q"]):
            error = actual - logit(q)
            losses.append(2 * max(tau * error, (tau - 1) * error))
    return sum(losses) / len(losses)


def answer_for(case, phi):
    latest = case["history"][-1]
    old_players, new_players = latest["players_tracked"], 3868 - latest["players_tracked"]
    answer = []
    for rate in latest["rates"]:
        if rate is None:
            answer.append(0.01)
            continue
        old_count = rate * old_players / 100
        new_count = new_players * rate / 100 * phi
        answer.append(100 * (old_count + new_count) / 3868)
    return answer


def coverage(answer, rows):
    return sum(row["q"][0] <= value <= row["q"][4] for value, row in zip(answer, rows)) / len(answer)


def main():
    cases = [load_json(CASES / f"p{index:02}.in") for index in range(2)]
    reports = {}
    for case in cases:
        case_id = case["id"]
        reports[case_id] = {}
        for name, program in (("entry", ROOT / "main.py"), ("starter", STARTER)):
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                first, result, elapsed = run(program, case, work)
                (work / "output" / f"{case_id}.out").unlink()
                second, _, elapsed2 = run(program, case, work)
                if first != second:
                    raise AssertionError(f"{name} {case_id} is nondeterministic")
                reports[case_id][name] = (result["trophies"], max(elapsed, elapsed2))
    print("Practice validation: 55 rows, five quantiles, bounds/order, <=65536 bytes")
    print("case  program  deterministic  runtime(s)  bytes")
    for case in cases:
        for name in ("entry", "starter"):
            _, seconds = reports[case["id"]][name]
            with tempfile.TemporaryDirectory() as directory:
                raw, _, _ = run(ROOT / "main.py" if name == "entry" else STARTER, case, Path(directory))
            print(f"{case['id']:4}  {name:7}  yes            {seconds:9.4f}  {len(raw)}")
    print("\nSynthetic refresh scoring (mean logit pinball; 90% coverage)")
    print("case  phi   entry loss/coverage   starter loss/coverage")
    for case in cases:
        for phi in (0, 0.1, 0.25, 0.5):
            answer = answer_for(case, phi)
            entry_rows = reports[case["id"]]["entry"][0]
            base_rows = reports[case["id"]]["starter"][0]
            print(f"{case['id']:4}  {phi:>4.2f}  {pinball(answer, entry_rows):.6f}/{coverage(answer, entry_rows):.3f}          {pinball(answer, base_rows):.6f}/{coverage(answer, base_rows):.3f}")
    case = json.loads(json.dumps(cases[0]))
    case["id"] = "synthetic-final"
    case["horizon_minutes"] = 6510
    case["target"] = "2026-11-26T00:00:00Z"
    case["at"] = "2026-11-21T11:30:00Z"
    case["history"] = []
    from datetime import datetime, timedelta, timezone

    for step, hour in enumerate((0, 24, 48)):
        dt = datetime(2026, 11, 19, tzinfo=timezone.utc) + timedelta(hours=hour)
        case["history"].append({"slot": dt.isoformat().replace("+00:00", "Z"), "at": dt.isoformat().replace("+00:00", "Z"), "players_tracked": 1000 + step * 1000, "trophies_earned": 0, "rates": [min(80, rate * (1 + step * 0.25)) if rate is not None and rate < 99 else rate for rate in cases[0]["history"][-1]["rates"]]})
    with tempfile.TemporaryDirectory() as directory:
        _, final_result, _ = run(ROOT / "main.py", case, Path(directory))
    final_rates = case["history"][-1]["rates"]
    rising = [i for i, rate in enumerate(final_rates) if rate is not None and rate < 90]
    above = sum(final_result["trophies"][i]["q"][2] > final_rates[i] for i in rising)
    widths = [logit(row["q"][4]) - logit(row["q"][0]) for row in final_result["trophies"] if row["id"] != 2]
    spread = sum(widths) / len(widths) / (2 * 1.645)
    print(f"\nSynthetic final-like: rising medians {above}/{len(rising)} above last rates; mean implied spread={spread:.3f} logit (target about 1.0)")


if __name__ == "__main__":
    main()
