import json
import pandas as pd

BS = chr(92)  # backslash

with open("data/specialist/model3/split_indices.json") as f:
    split = json.load(f)

for p in split["train"][:500]:
    p_norm = str(p).replace(BS, "/")
    if "/cleaned/tomato_foliar_spot/" in p_norm and "/recomp" not in p_norm:
        print("Split path:", repr(p))
        print("Normalized:", p_norm)
        break
else:
    print("No lab foliar found in first 500")

df = pd.read_csv("data/specialist/model3/mask_precompute_log.csv")
match = df[df["class_name"] == "tomato_foliar_spot"].iloc[0]
print("Mask log path:", match.image_path)
print("Mask log fg_path:", match.fg_path)
ml_norm = str(match.image_path).replace(BS, "/")
print()
print("Does p_norm endswith ml_norm?", p_norm.endswith(ml_norm))
print("Is ml_norm in p_norm?", ml_norm in p_norm)
# Try a partial match
ml_tail = "/".join(ml_norm.split("/")[-3:])
print(f"ml_tail = {ml_tail!r}")
print("Does p_norm endswith ml_tail?", p_norm.endswith(ml_tail))
