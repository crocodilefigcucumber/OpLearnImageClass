import pandas as pd
import subprocess
import glob

# PATH may need to be set
PATH = ""
resolutions = [13, 18, 23, 33, 38, 43, 48, 53, 58]

for res in resolutions:
    df = pd.read_csv("vit_eval_acc.csv")

    # Use glob to expand the checkpoint path wildcard
    ckpt_paths = glob.glob(f"{PATH}output/default/full_final/checkpoints/best*")

    if ckpt_paths:  # Ensure at least one matching checkpoint is found
        command = [
            "python",
            "main.py",
            "validate",
            "--config",
            f"{PATH}output/default/full_final/config.yaml",
            "--ckpt_path",
            ckpt_paths[
                0
            ],  # Use the first matching checkpoint, there's hopefully only one so doesn't matter ;)
            "--data.root",
            f"data/FMNIST_RGB_{res}_trigo",
        ]

        # Run the command
        result = subprocess.run(command, capture_output=False)
        result = pd.read_csv("./output/default/version_0/metrics.csv")["val_acc"].item()
        print(res, result)
        df.loc[len(df)] = {"res": res, "acc": result}
        df.to_csv("vit_eval_acc.csv", index=False)

    else:
        print(
            f"No checkpoint files found for pattern: {PATH}/output/default/full_final/checkpoints/best*"
        )
