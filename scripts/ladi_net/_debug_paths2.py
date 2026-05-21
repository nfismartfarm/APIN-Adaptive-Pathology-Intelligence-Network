import json
import pandas as pd

BS = chr(92)
with open("data/specialist/model3/split_indices.json") as f:
    split = json.load(f)

df = pd.read_csv("data/specialist/model3/mask_precompute_log.csv")
mask_paths_set = set(df["image_path"].map(lambda p: str(p).replace(BS, "/")))

# Pick a specific lab-looking split path and check if it's in mask_paths_set
tested = 0
found = 0
for p in split["train"][:1000]:
    p_norm = str(p).replace(BS, "/")
    if "/cleaned/tomato_foliar_spot/" in p_norm and "/recomp" not in p_norm:
        tested += 1
        # Try exact suffix match
        matched = any(p_norm.endswith(mp) for mp in mask_paths_set if "tomato_foliar_spot" in mp)
        if matched:
            found += 1
        else:
            # Debug: show the filename and see if a matching filename exists in mask_paths
            filename = p_norm.split("/")[-1]
            candidates = [mp for mp in mask_paths_set if mp.endswith(filename)]
            if candidates:
                print(f"FILENAME MATCH but path suffix differs:")
                print(f"  split: {p_norm}")
                print(f"  mask:  {candidates[0]}")
                break
            else:
                print(f"NO FILENAME MATCH: split filename {filename!r} not in mask log")
                break
print(f"Tested {tested} lab foliar paths; {found} matched via endswith")

# Also test LAB image count in split
lab_type_split = sum(
    1 for p in split["train"]
    if "/cleaned/" in str(p).replace(BS, "/")
    and "/recomp" not in str(p).replace(BS, "/")
    and not str(p).replace(BS, "/").split("/")[-1].startswith("recomp_")
)
print(f"Lab-looking paths in split['train']: {lab_type_split}")
print(f"Total mask log rows: {len(df)}")
