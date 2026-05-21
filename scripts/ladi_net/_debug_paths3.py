import json, pandas as pd
from pathlib import Path

BS = chr(92)
with open("data/specialist/model3/split_indices.json") as f:
    split = json.load(f)

df = pd.read_csv("data/specialist/model3/mask_precompute_log.csv")

# Find a lab image that IS in the mask log (plantvillage) and appears in split
mask_paths = df["image_path"].map(lambda p: str(p).replace(BS, "/")).tolist()
mask_set = set(mask_paths)

# Look for a plantvillage tomato image in the split
for i, p in enumerate(split["train"][:5000]):
    p_norm = str(p).replace(BS, "/")
    if "plantvillage_tomato" in p_norm and "/cleaned/" in p_norm:
        # Check if the tail matches
        fname = p_norm.split("/")[-1]
        candidates = [mp for mp in mask_paths if mp.endswith(fname)]
        if candidates:
            print(f"Split path {i}: {p_norm}")
            print(f"Mask entry:     {candidates[0]}")
            print(f"endswith works? {p_norm.endswith(candidates[0])}")
            # Check fg_path exists
            fg_row = df[df["image_path"].map(lambda x: str(x).replace(BS, "/")) == candidates[0]].iloc[0]
            fg = str(fg_row.fg_path)
            print(f"fg_path (csv):  {fg}")
            print(f"fg exists?      {Path(fg).exists()}")
            # Try the absolute-path version
            proj = Path(".").resolve()
            fg_abs = proj / fg.replace("/", BS) if BS in str(proj) else proj / fg
            print(f"fg abs:         {fg_abs}")
            print(f"fg abs exists?  {fg_abs.exists()}")
            break

# Count how many split train paths CONTAIN a mask_log filename anywhere
n_has_mask_name = 0
mask_names = set(Path(mp).name for mp in mask_paths)
for p in split["train"]:
    if Path(str(p)).name in mask_names:
        n_has_mask_name += 1
print(f"\nSplit train paths whose filename appears in mask_log: {n_has_mask_name}")
