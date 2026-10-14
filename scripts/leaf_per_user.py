"""
Split LEAF JSON files (several users per file) into one file per user, the layout the original Plexus code expects
for its training directory (each simulated node reads its share of the files). COMPASS-FL reads either layout.

    python scripts/leaf_per_user.py ~/leaf/femnist/data/train_raw ~/leaf/femnist/data/train
"""
import json
import os
import sys


def main(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    count = 0
    for name in sorted(os.listdir(src)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(src, name)) as handle:
            content = json.load(handle)
        for user in content["users"]:
            data = content["user_data"][user]
            with open(os.path.join(dst, user + ".json"), "w") as out:
                json.dump({"users": [user], "num_samples": [len(data["y"])], "user_data": {user: data}}, out)
            count += 1
    print("Wrote %d user files to %s" % (count, dst))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(os.path.expanduser(sys.argv[1]), os.path.expanduser(sys.argv[2]))
