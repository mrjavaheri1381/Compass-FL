"""
Split LEAF JSON files (several users per file) into one file per user, the layout the original Plexus code expects
for its training directory (each simulated node reads its share of the files).

Layout under ~/leaf/<dataset>/ (or --dataset-base-path/leaf/<dataset>/):
    data/train/            LEAF output, several users per file   (COMPASS-FL)
    data/test/             LEAF output                           (COMPASS-FL and the original Plexus code)
    per_user_data/train/   one file per user, made by this script (the original Plexus code)

    python scripts/leaf_per_user.py ~/leaf/femnist/data/train ~/leaf/femnist/per_user_data/train --min-samples 20

--min-samples drops users with fewer training samples: the original Plexus trainer uses drop_last=True with batch size
20, so such a user has no batch and its training crashes (StopIteration).
"""
import json
import os
import sys


def main(src: str, dst: str, min_samples: int = 0) -> None:
    os.makedirs(dst, exist_ok=True)
    count = dropped = 0
    for name in sorted(os.listdir(src)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(src, name)) as handle:
            content = json.load(handle)
        for user in content["users"]:
            data = content["user_data"][user]
            if len(data["y"]) < min_samples:
                dropped += 1
                continue
            with open(os.path.join(dst, user + ".json"), "w") as out:
                json.dump({"users": [user], "num_samples": [len(data["y"])], "user_data": {user: data}}, out)
            count += 1
    print("Wrote %d user files to %s (%d users with fewer than %d samples dropped)" % (count, dst, dropped, min_samples))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--min-samples", type=int, default=0)
    args = parser.parse_args()
    main(os.path.expanduser(args.src), os.path.expanduser(args.dst), args.min_samples)
