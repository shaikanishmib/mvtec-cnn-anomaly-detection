"""Add clone-safe text-line-ending checks without altering original snapshot hashes."""
import hashlib
import json
from pathlib import Path
from anomaly_detection.efficientad_final import ROOT,REPORT,atomic_json
from anomaly_detection.data import sha256


def main():
    path=REPORT/'baseline_preservation.json';baseline=json.loads(path.read_text())
    if 'canonical_lf_sha256' in baseline:
        print('Baseline line-ending qualification already retained');return
    canonical={}
    for name,expected in baseline['files'].items():
        file=ROOT/name
        assert sha256(file)==expected,'Earlier file changed; cannot qualify snapshot: '+name
        if file.suffix in ('.py','.md','.json','.csv','.txt','.toml','.ipynb') or file.name in ('.gitattributes','.gitignore','LICENSE'):
            canonical[name]=hashlib.sha256(file.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    baseline['canonical_lf_sha256']=canonical
    baseline['line_ending_note']='Original byte hashes retained. Canonical LF fallback permits Git text checkout normalization only; all original bytes verified before adding fallback. Experimental values/source content cannot change.'
    atomic_json(path,baseline)
    print('Qualified',len(canonical),'text files after verifying all original snapshot byte hashes')


if __name__=='__main__':main()
