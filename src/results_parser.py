import json
import glob
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CNN_RESULTS_PATH = PROJECT_ROOT / "outputs" / "01_baseline" / "results.jsonl"
YOLO_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "02_yolo"

def get_cnn_results() -> pd.DataFrame:
    """Lee y procesa los resultados de los experimentos CNN de results.jsonl."""
    if not CNN_RESULTS_PATH.exists():
        return pd.DataFrame()
        
    PAPER_REFERENCE = {
        'Segmented RGB': 96.18,
        'Canonical aligned': 96.18,
        'Raw (full image)': 83.54, 
    }

    rows = []
    with open(CNN_RESULTS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                exp_name = r.get('experiment', '')
                
                ref = None
                for key, val in PAPER_REFERENCE.items():
                    if exp_name.startswith(key):
                        ref = val
                        break
                rows.append({
                    'timestamp': r.get('timestamp', ''),
                    'experimento': r.get('experiment', ''),
                    'augment': 'Sí' if r.get('augment') else 'No',
                    'runs': r.get('n_runs', 0),
                    'épocas': r.get('epochs', 0),
                    'mean (%)': round(r.get('mean_acc', 0) * 100, 2),
                    'std (%)':  round(r.get('std_acc', 0) * 100, 2),
                    'paper (%)': ref if ref else None,
                    'nota': r.get('note', ''),
                })
            except json.JSONDecodeError:
                continue
                
    return pd.DataFrame(rows)

def get_best_cnn_per_setup() -> pd.DataFrame:
    """Retorna la corrida más robusta (con más runs) por cada setup experimental."""
    df = get_cnn_results()
    if df.empty:
        return df
    
    # Agrupar por experimento y augment, quedarse con el que tenga más 'runs'
    idx = df.groupby(['experimento', 'augment'])['runs'].idxmax()
    best_df = df.loc[idx].sort_values(by=['mean (%)'], ascending=True).reset_index(drop=True)
    return best_df

def get_yolo_results() -> pd.DataFrame:
    """Lee todos los archivos eval_*.json de la carpeta 02_yolo y devuelve un DataFrame comparativo."""
    if not YOLO_OUTPUTS_DIR.exists():
        return pd.DataFrame()
        
    eval_files = glob.glob(str(YOLO_OUTPUTS_DIR / "eval_*.json"))
    rows = []
    for path in eval_files:
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                run_name = data.get('run_name', Path(path).stem.replace('eval_', ''))
                
                rows.append({
                    'Modelo': run_name,
                    'Detection rate': data.get('detection_rate', 0) * 100,
                    'Misses': data.get('misses', 0),
                    'Acc global (todas)': data.get('classification_accuracy', 0) * 100,
                    'Acc (solo detectadas)': data.get('classification_accuracy_detected_only', 0) * 100,
                    'IoU promedio': data.get('mean_iou', 0),
                    'IoU mediana': data.get('median_iou', 0),
                    'mAP@50': data.get('mAP50', 0),
                    'mAP@50-95': data.get('mAP50_95', 0),
                    'Precision': data.get('precision', 0),
                    'Recall': data.get('recall', 0),
                })
            except json.JSONDecodeError:
                continue
            
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by='Acc global (todas)', ascending=False).reset_index(drop=True)
    return df
