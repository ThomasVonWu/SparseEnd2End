import os
import argparse
import subprocess

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="E2E Offline Evaluation ")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--iter_per_epoch", type=int, default=0)
    parser.add_argument(
        "--dir_path", type=str, default="", help="training checkpoint cache dir path."
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    for i in tqdm(range(151)):
        ids = args.iter_per_epoch * i
        ckpts = "iter_" + str(ids) + ".pth"
        ckpt_file = args.dir_path + ckpts
        if not os.path.exists(ckpt_file):
            print("ERROR")
            break
        with open(f"e2e_worklog/offline_evaluation_epoch{i}.log", "w") as log_file:
            subprocess.run(
                [
                    "python",
                    "script/test.py",
                    args.cfg,
                    "--checkpoint",
                    ckpt_file,
                ],
                stdout=log_file,
            )


if __name__ == "__main__":
    main()
