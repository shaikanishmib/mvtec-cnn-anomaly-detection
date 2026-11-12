"""Build the initial executable research notebook; never invent cell outputs."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
n = nbf.v4.new_notebook()
n.metadata.kernelspec = dict(display_name="MVTec research (CUDA, Python 3.12)", language="python", name="mvtec-research")
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
n.cells = [md("""# Notebook 02 — Convolutional Autoencoder

## Chronology: architecture and normal-only pilot
This notebook begins Milestone 4. Development selection and the final benchmark
are **pending**. Notebook 01 and its artifacts remain unchanged. No autoencoder
official test evaluation has occurred. All numeric outputs below are from execution,
not checkpoint replay. Full local images/checkpoints remain outside Git.

An autoencoder learns an encoder $z=f_\\theta(x)$ and decoder
$\\hat{x}=g_\\phi(z)$ that reproduce the input. Stride-two convolutions apply learned
local filters and halve spatial dimensions. The decoder doubles spatial dimensions
using bilinear interpolation and then learns convolutional filters. A linear 1x1
convolution creates the latent bottleneck; no skip connection bypasses it.

For a 256-square RGB input, an 8x32x32 bottleneck contains 8,192 values versus
196,608 input values (24-fold reduction in scalar count; this is not a bit-rate
or information-theoretic guarantee). L1 loss averages $|x-\\hat{x}|$ over batch,
channels and pixels; MSE averages $(x-\\hat{x})^2$ and emphasizes large residuals.
Training on fitting normals encourages reconstruction of normal structure.
Anomalies can nevertheless reconstruct well through shared local patterns or
excess capacity. Conversely, normal detail may reconstruct poorly. Blurring from
compression can produce residuals at normal edges; these are mechanisms to test,
not claims about unobserved final performance.

The predeclared development categories are bottle, cable, carpet and tile, chosen
for object/texture coverage without using official test performance. Bounded
ablation: shallow depth 3/latent 8/L1; depth 4; latent 2; latent 32; MSE.
One factor changes at a time. Combined loss, smoothing and a resolution change
require recorded development evidence first. Pilot: bottle, seed 2026, 256 pixels,
batch 16, Adam 0.001, three epochs, float32. No pretrained weights or augmentation.
Pilot epochs are a throughput check, not a selected final epoch budget.
"""), code("""from pathlib import Path
import sys, json, time, copy, traceback, subprocess
ROOT = Path.cwd()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from anomaly_detection.basic_cnn import seed_everything
from anomaly_detection.cnn_research import read_partition, require_fitting, load_resized
from anomaly_detection.data import sha256
from anomaly_detection.tracking import record
from anomaly_detection.autoencoder import ConvAutoencoder, reconstruction_loss, anomaly_map, image_scores
assert torch.cuda.is_available(), 'Pilot requires the existing CUDA environment'
torch.set_num_threads(4)
print(torch.__version__, torch.cuda.get_device_name(0))
"""), md("## Actual architecture and scoring implementation\nThe following source is executed directly in the notebook. Channel-mean absolute\nand squared residuals are separate candidate maps. A fixed top-1% mean aggregates each map into an image score; a maximum can be dominated by one noisy pixel.\nNeither score is a probability. No smoothing is applied.".replace('\\n+','\\n')),
code((root/'src/anomaly_detection/autoencoder.py').read_text()), code("""config = dict(width=16, depth=3, latent_channels=8, loss='l1', resolution=256,
              batch_size=16, learning_rate=.001, epochs=3, seed=2026)
identifier = '02-pilot-shallow-bottle-s2026-v1'
compact = ROOT / 'reports/autoencoder'
local = ROOT / 'outputs/autoencoder' / identifier
compact.mkdir(parents=True, exist_ok=True)
local.mkdir(parents=True, exist_ok=True)
checkpoint = ROOT / 'checkpoints/autoencoder' / identifier / 'last.pt'
checkpoint.parent.mkdir(parents=True, exist_ok=True)
entry = dict(experiment_id=identifier+'-start', model_family='convolutional_autoencoder',
    categories=['bottle'], hypothesis='A scratch bottleneck model learns to reconstruct fitting normals; measure throughput before development.',
    preprocessing='PIL RGB bilinear 256x256, float32 /255', hyperparameters=config, seed=2026,
    environment=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
      git_sha=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
      dirty=bool(subprocess.check_output(['git','status','--porcelain'], cwd=ROOT)),
      source_sha256=sha256(ROOT/'src/anomaly_detection/autoencoder.py'),
      lock_sha256=sha256(ROOT/'requirements.lock.txt'),
      partitions=json.loads((ROOT/'reports/dataset_validation.json').read_text())['partitions']),
    training={}, metrics={}, calibration={}, paths=dict(local=str(local.relative_to(ROOT))),
    interpretation='Throughput pilot only; no official test access.', decision='Measure before scaling', status='started')
record(ROOT/'experiments', entry)
seed_everything(config['seed'])
model = ConvAutoencoder(**{k:config[k] for k in ('width','depth','latent_channels')}).cuda()
print(model)
print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
with torch.no_grad():
    x = torch.zeros(1,3,256,256,device='cuda')
    z = model.bottleneck(model.encoder(x))
    print('Input / latent / output:', x.shape, z.shape, model(x).shape)
"""), md("## Normal-only training loop\nOnly frozen fitting normals receive gradients. The reserved development normals\nmeasure reconstruction each epoch. Calibration and official test data are not loaded.\nAll three epochs are retained; the pilot performs no checkpoint selection."), code("""history = []
started = time.perf_counter()
try:
    fit_rows = read_partition('fit', 'bottle')
    require_fitting(fit_rows)
    dev_rows = read_partition('development', 'bottle')
    fit = load_resized(fit_rows, config['resolution']).cuda()
    dev = load_resized(dev_rows, config['resolution']).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    shuffle = torch.Generator().manual_seed(config['seed'])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, config['epochs']+1):
        tick = time.perf_counter()
        model.train()
        total = 0.
        for indices in torch.randperm(len(fit), generator=shuffle).split(config['batch_size']):
            batch = fit[indices.cuda()].float()/255
            optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_loss(model(batch), batch, config['loss'])
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            total += loss.item()*len(batch)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter()-tick
        model.eval()
        with torch.inference_mode():
            validation = sum(reconstruction_loss(model(b.float()/255), b.float()/255, config['loss']).item()*len(b)
                             for b in dev.split(config['batch_size']))/len(dev)
        torch.cuda.synchronize()
        row = dict(epoch=epoch, train_loss=total/len(fit), normal_development_loss=validation,
                   train_seconds=train_seconds, elapsed_seconds=time.perf_counter()-started,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
        history.append(row)
        pd.DataFrame(history).to_csv(compact/(identifier+'-history.csv'),index=False)
        print(json.dumps(row), flush=True)
    torch.save(dict(state_dict=model.state_dict(), configuration=config, category='bottle',
                    epoch=epoch, optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(), shuffle_rng=shuffle.get_state()), checkpoint)
    summary = dict(configuration=config, parameters=sum(p.numel() for p in model.parameters()),
                   fit_count=len(fit), development_count=len(dev), history=history,
                   wall_seconds=time.perf_counter()-started, checkpoint=str(checkpoint.relative_to(ROOT)),
                   checkpoint_sha256=sha256(checkpoint), checkpoint_bytes=checkpoint.stat().st_size)
    result = copy.deepcopy(entry)
    result.update(experiment_id=identifier+'-result', status='completed', metrics=summary,
                  decision='Pilot complete; development and freeze pending',
                  interpretation='See actual curves; no detection quality measured by reconstruction loss alone.')
    record(ROOT/'experiments', result)
    with (compact/(identifier+'-result.json')).open('x') as stream: json.dump(summary,stream,indent=2)
except BaseException as exc:
    (local/'failure.txt').write_text(traceback.format_exc())
    failed = copy.deepcopy(entry)
    failed.update(experiment_id=identifier+'-failure',status='failed',interpretation=str(exc),decision='Preserve and investigate')
    record(ROOT/'experiments',failed)
    raise
"""), code("""frame = pd.DataFrame(history)
frame.plot(x='epoch', y=['train_loss','normal_development_loss'], marker='o', ylabel='Mean absolute RGB error')
plt.show()
print(frame.to_string(index=False))
print('Mean GPU training seconds per epoch:', frame.train_seconds.mean())
print('Peak allocated GiB:', frame.peak_allocated_bytes.max()/2**30)
print('Illustrative 45-model, 20-epoch training seconds (before category-size adjustment):',
      frame.train_seconds.mean()*45*20)
"""), md("## Pilot reconstruction panels (local only)\nFirst three development images in manifest order; no favorable image selection.\nDisplay scales are fixed to [0,1]. These panels remain local under the dataset\nretention policy; the public notebook retains the measured curves and tables."), code("""with torch.inference_mode():
    originals = dev[:3].float()/255
    reconstructions = model(originals)
    absolute = anomaly_map(originals, reconstructions, 'absolute')
    squared = anomaly_map(originals, reconstructions, 'squared')
fig, axes = plt.subplots(3,4,figsize=(12,8))
for i in range(3):
    for j, values in enumerate([originals, reconstructions, absolute, squared]):
        a = values[i].cpu().numpy()
        axes[i,j].imshow(a.transpose(1,2,0) if a.ndim==3 else a, vmin=0,vmax=1)
        axes[i,j].axis('off')
for ax,title in zip(axes[0],['Normal original','Reconstruction','Absolute residual','Squared residual']): ax.set_title(title)
plt.tight_layout()
plt.show()
"""), md("""## Pending research and limits
The pilot does not measure detection or localization quality. Its three epochs
cannot establish convergence. A later append-only interpretation will use these
measured curves to set a bounded development epoch budget. Before final training,
publish an estimate adjusted for all category fitting counts, selected epoch count,
three seeds and measured memory. No all-category training has begun.

Development must compare the declared architectures, bottleneck capacities and
losses, plus absolute versus squared maps on fixed fitting-derived synthetic
probes and normal development evidence. Final policy, epochs, seeds, map and
aggregation must then be frozen before calibration/test evaluation. Image thresholds
will use the shared higher empirical 95th percentile of reserved normal scores,
with alarm strictly above threshold; pixel calibration needs its own rule.

All-category metrics, calibration counts/uncertainty, failure analysis and exact
stored Basic CNN comparisons are pending. No claim of improvement is made.
Experiment records: [append-only registry](../experiments/); pilot curves and
resource evidence: [reports](../reports/autoencoder/).
""")]
n.cells[-2].metadata['local_dataset_pixels'] = True
nbf.write(n, root/'notebooks/02_convolutional_autoencoder.ipynb')

if __name__ == '__main__':
    import os
    from nbclient import NotebookClient
    os.environ['JUPYTER_PLATFORM_DIRS'] = '1'
    os.environ['IPYTHONDIR'] = str(root/'outputs/autoencoder/ipython')
    NotebookClient(n, timeout=900, kernel_name='mvtec-research', resources={'metadata':{'path':str(root)}}).execute()
    local = root/'outputs/autoencoder/02_convolutional_autoencoder.executed.ipynb'
    nbf.write(n, local)
    for cell in n.cells:
        if cell.metadata.get('local_dataset_pixels'): cell.outputs = []
    nbf.write(n, root/'notebooks/02_convolutional_autoencoder.ipynb')
