"""Local inference command; no HTTP service or application framework."""
import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np
from .patchcore import PatchCorePredictor


def main():
    parser=argparse.ArgumentParser(description='Frozen simplified PatchCore inference')
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--artifact-root',type=Path,required=True)
    parser.add_argument('--category',required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--output-dir',type=Path,help='New directory for JSON, raw map and optional visualizations')
    parser.add_argument('--visualize',action='store_true')
    args=parser.parse_args()
    model=PatchCorePredictor(args.manifest,args.artifact_root,args.category,args.device,args.manifest_sha256)
    prediction=model.predict(args.input,visualize=args.visualize)
    response=prediction.to_dict()
    if args.output_dir:
        args.output_dir.mkdir(parents=True,exist_ok=False)
        (args.output_dir/'prediction.json').write_text(json.dumps(response,indent=2)+'\n')
        np.save(args.output_dir/'anomaly_map.npy',prediction.anomaly_map,allow_pickle=False)
        if args.visualize:
            Image.fromarray(prediction.heatmap).save(args.output_dir/'heatmap.png')
            Image.fromarray(prediction.overlay).save(args.output_dir/'overlay.png')
    print(json.dumps(response,indent=2))


if __name__=='__main__':main()
