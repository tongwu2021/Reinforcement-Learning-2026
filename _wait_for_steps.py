import time
import wandb

TARGET_STEPS = 1_950_000  # current(~1,450,000) + 500,000
VALIDATION_INTERVAL = 50_000

run_path = "le993939-korea-university-of-technology-and-education/sac_BipedalWalkerHardcore-v3/egibo1gz"

# NOTE: "[TRAIN] Replay buffer" saturates at the buffer capacity (1,000,000), so it can no
# longer be used as a step proxy past that point. Each validation/log event fires exactly
# every VALIDATION_INTERVAL steps, so (number of logged rows) * VALIDATION_INTERVAL is an
# accurate estimate of the current time_steps.
while True:
    api = wandb.Api()
    run = api.run(run_path)
    hist = run.history(samples=2000, keys=["[TRAIN] alpha"])
    latest = len(hist) * VALIDATION_INTERVAL
    print(f"latest time_steps ~= {latest:,} (target {TARGET_STEPS:,})", flush=True)
    if latest >= TARGET_STEPS:
        print("TARGET REACHED", flush=True)
        break
    time.sleep(180)
