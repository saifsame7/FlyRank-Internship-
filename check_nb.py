import json
try:
    nb = json.load(open('work/notebooks/w04_baseline_score.ipynb'))
    print("Loaded successfully")
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = cell.get('source', [])
            print(f"Cell {i} source type: {type(src)}, len: {len(src) if isinstance(src, list) else 'N/A'}")
            for j, s in enumerate(src[:5]):
                print(f"  source[{j}]: {repr(s[:60])}")
except Exception as e:
    print(f"Error: {e}")