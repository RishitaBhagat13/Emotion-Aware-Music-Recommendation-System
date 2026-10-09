import numpy as np
import json

# Load labels used during training
y = np.load("training/y.npy")

classes = sorted(list(set(y)))

label_map = {
    int(i): int(c)
    for i, c in enumerate(classes)
}

with open(
    "training/label_map.json",
    "w"
) as f:
    json.dump(
        label_map,
        f,
        indent=4
    )

print("Saved label map")
print("Classes:", len(classes))

for i in range(10):
    print(
        i,
        "->",
        label_map[i]
    )