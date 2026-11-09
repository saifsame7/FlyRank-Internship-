import json

for fname in ['work/notebooks/w03_data_contract.ipynb', 'work/notebooks/w03_feature_leakage_check.ipynb']:
    with open(fname) as f:
        nb = json.load(f)
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown' and isinstance(cell.get('source'), str):
            lines = cell['source'].split('\\n')
            cell['source'] = [l for l in lines if l]
    with open(fname, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'Rewrote {fname} with list-of-strings source')