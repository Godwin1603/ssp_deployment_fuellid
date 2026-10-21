import os

def replace_in_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

app_replacements = [
    ('"instruction": "WAITING FOR PART"', '"instruction": "WAITING FOR DOOR"'),
    ('"instruction": "OK - REMOVE THE PLATE"', '"instruction": "OK - REMOVE THE DOOR"'),
    ('"instruction": "NG - REMOVE THE PLATE"', '"instruction": "NG - REMOVE THE DOOR"'),
    ('"status": "Part Out of Bounds!"', '"status": "Door Out of Bounds!"'),
    ('"status": "Part Type Mismatch!"', '"status": "Door Type Mismatch!"'),
    ('"instruction": "PLACE CORRECT PART"', '"instruction": "PLACE CORRECT DOOR"'),
    ('"status": "Part removed. Saving report..."', '"status": "Door removed. Saving report..."'),
]

index_replacements = [
    ('ADJUST PART — TOUCHING ROI BOUNDARY', 'ADJUST DOOR — TOUCHING ROI BOUNDARY'),
    ('WAITING FOR PART', 'WAITING FOR DOOR'),
    ('Part Status & Defects', 'Door Status & Defects'),
    ('how far (%) the part can slide', 'how far (%) the door can slide'),
    ('"PART OUT OF BOUNDS"', '"DOOR OUT OF BOUNDS"'),
    ('⚠ ADJUST THE PART ⚠', '⚠ ADJUST THE DOOR ⚠'),
    ("speech: 'Adjust the part. It is out of bounds'", "speech: 'Adjust the door. It is out of bounds'")
]

replace_in_file('app.py', app_replacements)
replace_in_file('index.html', index_replacements)
