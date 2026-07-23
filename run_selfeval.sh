set -e
PY=/c/Users/ritayan/miniconda3/python.exe
echo "=== [1/4] leakage audit + 2s metrics (both modes) ==="
$PY -u run.py --mode both --poses 360 --frames 120 --fps 60
echo "=== [2/4] 4s decoupled headline ==="
$PY -u run.py --mode decoupled --poses 600 --frames 240 --fps 60 --target-frequency 0.95 --target-severity 0.95
echo "=== [3/4] retrain export models ==="
$PY -u train_export.py --poses 500 --frames 180 --fps 60 --subjects 28
echo "=== [4/4] regenerate report ==="
$PY -u make_report.py
echo "ALL_DONE"
