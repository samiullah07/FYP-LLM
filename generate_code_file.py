import os
from datetime import datetime

ROOT = r"C:\Users\BEST LAPTOP\Desktop\FYP-LLM"

INCLUDE_EXT = {'.py', '.json', '.toml', '.txt', '.md', '.cfg'}

SKIP_DIRS = {
    '.venv', '__pycache__', '.git', '.claude',
    'node_modules', '.pytest_cache', 'bandit_log', 'media'
}

SKIP_FILES = {
    'uv.lock', '.python-version', 'ALL_PROJECT_CODE.txt',
    'poetry.lock', 'package-lock.json', 'generate_code_file.py'
}

collected = []

for dirpath, dirs, files in os.walk(ROOT):
    dirs[:] = sorted([d for d in dirs
                      if d not in SKIP_DIRS and not d.startswith('.')])
    for fname in sorted(files):
        ext = os.path.splitext(fname)[1]
        if ext not in INCLUDE_EXT:
            continue
        if fname in SKIP_FILES:
            continue
        full = os.path.join(dirpath, fname)
        rel  = os.path.relpath(full, ROOT)
        rel  = rel.replace('\\', '/')
        try:
            with open(full, 'r', encoding='utf-8', errors='replace') as f:
                code = f.read()
        except Exception as e:
            code = '[ERROR: ' + str(e) + ']'
        collected.append((rel, code))

OUT   = os.path.join(ROOT, 'ALL_PROJECT_CODE.txt')
WIDTH = 116

with open(OUT, 'w', encoding='utf-8') as out:
    out.write('=' * WIDTH + '\n')
    out.write('FYP-LLM — Agentic Academic Literature Review System\n')
    out.write('Complete Project Codebase Export\n')
    out.write('Generated : ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
    out.write('Files     : ' + str(len(collected)) + '\n')
    out.write('=' * WIDTH + '\n')

    for rel, code in collected:
        label  = '  ' + rel + '  '
        dashes = WIDTH - len(label) - 2
        left   = dashes // 2
        right  = dashes - left
        div    = '<' + '-' * left + label + '-' * right + '>'
        out.write('\n\n' + div + '\n\n')
        out.write(code)
        if not code.endswith('\n'):
            out.write('\n')

    out.write('\n\n' + '=' * WIDTH + '\n')
    out.write('END OF FILE — ' + str(len(collected)) + ' files combined\n')
    out.write('=' * WIDTH + '\n')

kb = os.path.getsize(OUT) // 1024
print('Done!')
print('Output : ' + OUT)
print('Size   : ' + str(kb) + ' KB')
print('Files  : ' + str(len(collected)))
print('')
for rel, code in collected:
    lines = code.count('\n')
    print('  ' + str(lines).rjust(5) + ' lines  ' + rel)
