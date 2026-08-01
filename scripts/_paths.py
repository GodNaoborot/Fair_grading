"""Общие пути проекта. Импортируется первым любым скриптом из scripts/."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TRACES = RESULTS / "traces"
PLOTS = RESULTS / "plots"
LOGS = RESULTS / "logs"

# Windows-консоль часто работает в cp1251 и падает с UnicodeEncodeError на
# символах вроде «×» или «→». Заменяем их вместо того, чтобы ронять скрипт.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="replace")

# Делаем src/ импортируемым (чтобы работало `from utils import ...`)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Создаём выходные директории, если их ещё нет
for d in (TRACES, PLOTS, LOGS):
    d.mkdir(parents=True, exist_ok=True)
