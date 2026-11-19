import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)


def clamp_rate(value):
    return min(99.99, max(0.01, value))


def logit(rate):
    rate = clamp_rate(rate)
    return math.log(rate / (100.0 - rate))


def logistic(value):
    return 100.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))


def fit_line(xs, ys, weights):
    total = sum(weights)
    mean_x = sum(w * x for w, x in zip(weights, xs)) / total
    mean_y = sum(w * y for w, y in zip(weights, ys)) / total
    denominator = sum(w * (x - mean_x) ** 2 for w, x in zip(weights, xs))
    if denominator < 1e-12:
        return mean_y, 0.0, 0.0
    slope = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(weights, xs, ys)) / denominator
    residual = math.sqrt(sum(w * (y - mean_y - slope * (x - mean_x)) ** 2 for w, x, y in zip(weights, xs, ys)) / total)
    return mean_y, slope, residual


def forecast_case(case):
    history = case["history"]
    target = datetime.fromisoformat(case["target"].replace("Z", "+00:00"))
    release = datetime.fromisoformat(case["game"]["release_date"] + "T00:00:00+00:00")
    target_x = math.log1p((target - release).total_seconds() / 3600.0)
    xs = [math.log1p((datetime.fromisoformat(row["at"].replace("Z", "+00:00")) - release).total_seconds() / 3600.0) for row in history]
    weights = [math.exp(2.0 * (x - xs[-1])) for x in xs]
    horizon_days = case["horizon_minutes"] / 1440.0
    walk_spread = 0.50 * math.sqrt(horizon_days)
    rows = []
    for index, trophy in enumerate(case["trophies"]):
        samples = [(x, row["rates"][index], row["players_tracked"]) for x, row in zip(xs, history) if row["rates"][index] is not None]
        rates = [sample[1] for sample in samples]
        if rates and all(rate == 100 for rate in rates):
            median, spread = logit(99.99), 0.05
        elif not samples:
            median, spread = logit(0.01), math.hypot(walk_spread, 1.0)
        else:
            sx = [sample[0] for sample in samples]
            sw = [math.exp(2.0 * (x - xs[-1])) for x in sx]
            ylogit = [logit(sample[1]) for sample in samples]
            _, slope, residual = fit_line(sx, ylogit, sw)
            trend = sum(w * y for w, y in zip(sw, ylogit)) / sum(sw) + slope * (target_x - sum(w * x for w, x in zip(sw, sx)) / sum(sw))
            count_logs = [math.log(max(0.01, sample[1] * sample[2] / 100.0)) for sample in samples]
            player_logs = [math.log(sample[2]) for sample in samples]
            count_mean, count_slope, _ = fit_line(sx, count_logs, sw)
            player_mean, player_slope, _ = fit_line(sx, player_logs, sw)
            count_forecast = count_mean + count_slope * (target_x - sum(w * x for w, x in zip(sw, sx)) / sum(sw))
            player_forecast = player_mean + player_slope * (target_x - sum(w * x for w, x in zip(sw, sx)) / sum(sw))
            count_rate = clamp_rate(100.0 * math.exp(min(700, count_forecast - player_forecast)))
            median = (trend + logit(count_rate)) / 2.0 - 1.05 * math.exp(-horizon_days / 2.0)
            residual = min(1.0, max(0.12, residual))
            refresh = 0.90 * math.exp(-horizon_days / 4.0)
            spread = math.sqrt(walk_spread ** 2 + residual ** 2 + refresh ** 2)
        quantiles = [round(clamp_rate(logistic(median + z * spread)), 4) for z in Z]
        for qi in range(1, len(quantiles)):
            quantiles[qi] = max(quantiles[qi], quantiles[qi - 1])
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
