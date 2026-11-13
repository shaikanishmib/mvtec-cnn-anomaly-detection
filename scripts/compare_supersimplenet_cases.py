"""Execute a matched frozen-prediction analysis, without CNN replay or fitting."""
from pathlib import Path
import os,asyncio,json,sys,time
import nbformat as nbf
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from anomaly_detection.tracking import record
from anomaly_detection.data import sha256
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
n=nbf.v4.new_notebook();n.metadata.kernelspec=dict(display_name='MVTec research (CUDA, Python3.12)',language='python',name='mvtec-research')
n.cells=[md('''# Matched frozen cases — additional artifact-analysis session
**Research question:** how do the four completed models respond to the same known difficult cases? Select four paths from the earlier PatchCore failure record: two texture normals, a manipulated screw tip and a misplaced transistor. These cases are illustrative, not a representative quality estimate. All45 SuperSimpleNet fits and official evaluations are already complete. No weights, image-score policy or threshold changes are permitted.

The following live analysis reads retained predictions and actual frozen thresholds; it does not perform new CNN feature extraction, training or model inference. Scores have different units across models; only the sign of each model's own threshold margin has common decision meaning. Heatmap scales are per-panel. Green contours mark native ground truth and cyan contours mark the model's frozen pixel decision. Original-image panels remain local; the public notebook retains genuine compact numbers and source.
'''),code('''from pathlib import Path
import sys,json
ROOT=Path.cwd()
if ROOT.name=='notebooks':ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch,matplotlib.pyplot as plt
from PIL import Image
from anomaly_detection.data import autodiscover_data_root,sha256
from anomaly_detection.cnn_research import read_partition
torch.set_num_threads(4)
families={'Basic CNN':'basic_cnn','Autoencoder':'autoencoder','PatchCore':'patchcore','SuperSimpleNet Lite':'supersimplenet'}
paths=['leather/test/good/001.png','carpet/test/good/003.png','screw/test/manipulated_front/000.png','transistor/test/misplaced/002.png']
data=autodiscover_data_root(ROOT);assets={};records=[]
for path in paths:
    category=path.split('/')[0];source=read_partition('test',category).set_index('path').loc[path]
    assert sha256(data/path)==source.sha256
    with Image.open(data/path) as image:rgb=np.asarray(image.convert('RGB'))
    if isinstance(source['mask'],str):
        assert sha256(data/source['mask'])==source.mask_sha256
        with Image.open(data/source['mask']) as image:truth=np.asarray(image)>0
    else:truth=np.zeros(rgb.shape[:2],bool)
    assets[path]={'rgb':rgb,'truth':truth,'models':{}}
    for name,folder in families.items():
        runs=pd.read_csv(ROOT/f'reports/{folder}/all_runs.csv').set_index(['category','seed'])
        identifier=runs.loc[(category,2026),'experiment_id']
        result=json.loads((ROOT/f'reports/{folder}/evaluation/{identifier}-result.json').read_text())
        images=pd.read_csv(ROOT/f'reports/{folder}/evaluation/{identifier}-images.csv')
        index=int(images.index[images.path.eq(path)][0]);row=images.iloc[index]
        assert sha256(ROOT/result['local_maps'])==result['maps_sha256']
        small=np.load(ROOT/result['local_maps'],mmap_mode='r')[index]
        native=torch.nn.functional.interpolate(torch.from_numpy(np.array(small))[None,None],size=truth.shape,mode='bilinear',align_corners=False).numpy()[0,0]
        pred=native>result['calibration']['pixel_threshold']
        assets[path]['models'][name]=dict(map=native,pred=pred,margin=float(row.score-result['calibration']['image_threshold']))
        records.append(dict(path=path,model=name,seed=2026,image_score=float(row.score),image_threshold=result['calibration']['image_threshold'],
            image_margin=float(row.score-result['calibration']['image_threshold']),outcome=row.outcome,
            pixel_threshold=result['calibration']['pixel_threshold'],predicted_pixel_fraction=float(pred.mean()),
            defect_pixel_recall=float(pred[truth].mean()) if truth.any() else None))
table=pd.DataFrame(records)
table.to_csv(ROOT/'reports/supersimplenet/matched_frozen_cases.csv',index=False,lineterminator='\\n')
display(table)
'''),code('''fig,axes=plt.subplots(len(paths),6,figsize=(17,11))
for axs,path in zip(axes,paths):
    asset=assets[path];axs[0].imshow(asset['rgb']);axs[0].set_title(path,fontsize=7)
    axs[1].imshow(asset['truth'],cmap='gray');axs[1].set_title('Native ground truth')
    for ax,(name,values) in zip(axs[2:],asset['models'].items()):
        ax.imshow(values['map'],cmap='inferno')
        if asset['truth'].any():ax.contour(asset['truth'],levels=[.5],colors='lime',linewidths=.7)
        if values['pred'].any() and not values['pred'].all():ax.contour(values['pred'],levels=[.5],colors='cyan',linewidths=.5)
        ax.set_title(f"{name}\\nimage margin {values['margin']:.4f}",fontsize=8)
    for ax in axs:ax.axis('off')
fig.tight_layout();fig.savefig(ROOT/'outputs/supersimplenet/matched-cases-four-models.png',dpi=110);plt.show();plt.close(fig)
'''),md('''The table and local panels are a descriptive comparison on selected prior failures. They cannot establish superiority over the full benchmark. Image decisions and pixel decisions remain distinct; image misses can coexist with localized pixel responses. The all-category macro results and paired uncertainty remain the primary comparative evidence.''')]
n.cells[2].metadata['local_dataset_pixels']=True
for c in n.cells:c.metadata['execution_session']='matched-frozen-case-analysis'
out=ROOT/'outputs/supersimplenet';executed=out/'04_supersimplenet.matched_cases.executed.ipynb'
assert not executed.exists()
event=dict(experiment_id='04-matched-frozen-cases-v1-start',model_family='CrossModelFrozenAnalysis',categories=['leather','carpet','screw','transistor'],
    hypothesis='Compare the same previously documented difficult images across four frozen methods without new inference or selection',
    preprocessing='Read original RGB/masks and retained native-interpolated maps only',hyperparameters=dict(seed=2026,cases='four prior PatchCore failure paths; explicitly selected examples'),seed=2026,
    environment=dict(source_sha256=sha256(Path(__file__))),training=dict(phase='post-confirmation artifact analysis; no training'),metrics={},calibration=dict(policy='All recorded thresholds unchanged'),
    paths=dict(local_notebook=executed.relative_to(ROOT).as_posix()),interpretation='Selected illustrative cases are not a random benchmark sample',decision='Describe without retuning',status='started')
record(ROOT/'experiments',event);started=time.perf_counter()
os.environ['IPYTHONDIR']=str(out/'ipython')
if os.name=='nt':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
try:NotebookClient(n,timeout=900,kernel_name='mvtec-research',resources={'metadata':{'path':str(ROOT)}}).execute()
finally:executed.write_bytes(nbf.writes(n).encode())
event.update(experiment_id='04-matched-frozen-cases-v1-result',status='completed',metrics=dict(seconds=time.perf_counter()-started,
    table_sha256=sha256(ROOT/'reports/supersimplenet/matched_frozen_cases.csv'),executed_notebook_sha256=sha256(executed)),decision='Retain full selected-case comparison and unchanged benchmark')
record(ROOT/'experiments',event)
print('Matched analysis executed; append the actual session only after canonical notebook validation finishes')
