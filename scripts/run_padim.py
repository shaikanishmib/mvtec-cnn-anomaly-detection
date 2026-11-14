"""Explicit PaDiM gates; each completed result is retained before advancing."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import torch
from anomaly_detection.basic_cnn import seed_everything
from anomaly_detection.padim_research import pilot, development, freeze

if __name__=='__main__':
    torch.set_num_threads(4)
    seed_everything(2026)
    torch.backends.cuda.matmul.allow_tf32=False
    phase=sys.argv[1]
    if phase=='pilot':print(pilot())
    elif phase=='development':development()
    elif phase=='freeze':print(freeze())
    elif phase=='evaluate':
        from anomaly_detection.padim_evaluation import confirmation
        confirmation()
    else:raise ValueError('Unknown phase')
