import json

sign_map = {}

with open(
    "datasets/wlasl_class_list.txt",
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        parts = line.strip().split()

        if len(parts) >= 2:

            class_id = int(parts[0])

            if class_id < 100:

                sign_name = "_".join(parts[1:])

                sign_map[class_id] = sign_name

with open(
    "training/nslt100_sign_names.json",
    "w"
) as f:

    json.dump(
        sign_map,
        f,
        indent=4
    )

print("Saved", len(sign_map), "signs")

for i in range(20):
    print(i, "->", sign_map[i])