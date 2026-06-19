import json
nb = json.load(open(r"_phase0_sample_copy/notebooks/sync_free_eye_ellipse_pipeline.ipynb", encoding="utf-8"))
print("cells:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    print(f"--- cell {i} ({c['cell_type']}) ---")
    print("".join(c.get("source") or [""])[:1800])
