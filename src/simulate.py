import numpy as np
import pandas as pd
from utils import load_yaml, ensure_dir
from tqdm import tqdm

def simulate_session(df, lam, mu, sigma, servers, sla_list):
    # arrivals
    arrivals = np.cumsum(np.random.exponential(1/lam, size=len(df)))
    df = df.copy()
    df["arrival"] = arrivals

    # priority = -prob
    df["priority"] = -df["prob"]

    # simulate
    servers_free = np.zeros(servers)
    start = []
    end = []

    for _, row in df.sort_values("arrival").iterrows():
        t = row["arrival"]
        if np.min(servers_free) > t:
            t = np.min(servers_free)
        s = np.random.lognormal(mu, sigma)
        servers_free[np.argmin(servers_free)] = t + s
        start.append(t)
        end.append(t + s)

    df["start"] = start
    df["end"] = end
    df["ttr"] = df["end"] - df["arrival"]

    out = {}
    out["median_ttr"] = df["ttr"].median()
    for m in sla_list:
        out[f"sla_{m}"] = (df["ttr"] <= m).mean()*100
    return out

def main():
    cfg = load_yaml("configs/sim.yaml")

    df = pd.read_csv(cfg["PRED_CSV"])
    ensure_dir(cfg["OUTPUT_DIR"])

    results = []

    for s in tqdm(range(cfg["N_SESSIONS"])):
        out = simulate_session(
            df,
            cfg["LAMBDA_PER_HOUR"],
            cfg["SERVICE_LOGNORM_MU"],
            cfg["SERVICE_LOGNORM_SIGMA"],
            cfg["SERVERS"],
            cfg["SLA_MINUTES"]
        )
        out["session"] = s
        results.append(out)

    pd.DataFrame(results).to_csv(cfg["OUTPUT_DIR"] + "/sim_results.csv", index=False)

if __name__ == "__main__":
    main()
