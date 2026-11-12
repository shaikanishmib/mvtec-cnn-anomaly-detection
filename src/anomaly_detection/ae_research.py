"""Normal-only AE fits, with immutable experiment events and full learning curves."""
import copy
import json
import time
import traceback
from pathlib import Path
import torch
import pandas as pd
from sklearn.metrics import average_precision_score
from .autoencoder import ConvAutoencoder, reconstruction_loss, anomaly_map, image_scores
from .basic_cnn import seed_everything
from .cnn_research import ROOT, read_partition, require_fitting, load_resized, inner_fit_split, make_probes
from .data import sha256
from .tracking import record, config_hash
from .metrics import binary_metrics, calibrate_normal


@torch.inference_mode()
def reconstruct(model, images, batch_size=16):
    model.eval()
    return torch.cat([model(b.float()/255).cpu() for b in images.split(batch_size)])


def fit(identifier, category, config, final=False):
    compact = ROOT/'reports/autoencoder/runs'
    compact.mkdir(parents=True, exist_ok=True)
    local = ROOT/'outputs/autoencoder'/identifier
    local.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT/'checkpoints/autoencoder'/identifier/'last.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=False)
    entry = dict(experiment_id=identifier+'-start',model_family='ConvAutoencoder',categories=[category],
        hypothesis='Controlled normal-only reconstruction ablation' if not final else 'Frozen category-specific confirmation',
        preprocessing='PIL RGB bilinear, /255 float32', hyperparameters=config, seed=config['seed'],
        environment=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
            source_sha256={p:sha256(ROOT/p) for p in ['src/anomaly_detection/autoencoder.py','src/anomaly_detection/ae_research.py']},
            partitions=json.loads((ROOT/'reports/dataset_validation.json').read_text())['partitions'],
            lock_sha256=sha256(ROOT/'requirements.lock.txt')),
        training=dict(final=final),metrics={},calibration={},paths=dict(local=str(local.relative_to(ROOT))),
        interpretation='No calibration or official test access',decision='retain all epochs',status='started')
    record(ROOT/'experiments',entry)
    started=time.perf_counter()
    try:
        seed_everything(config['seed'])
        rows=read_partition('fit',category)
        require_fitting(rows)
        # Same fitting-derived held-out probes across variants; excluded from gradients during development.
        learning,probe_rows=(rows,None) if final else inner_fit_split(rows)
        images=load_resized(learning,config['resolution']).cuda()
        dev=load_resized(read_partition('development',category),config['resolution']).cuda()
        if not final:
            probe_images=load_resized(probe_rows,config['resolution']).cuda()
            probes,masks=make_probes(probe_images,probe_rows)
        model=ConvAutoencoder(**{k:config[k] for k in ['width','depth','latent_channels']}).cuda()
        optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
        shuffle=torch.Generator().manual_seed(config['seed'])
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        history=[]
        for epoch in range(1,config['epochs']+1):
            tick=time.perf_counter();model.train();total=0.
            for indices in torch.randperm(len(images),generator=shuffle).split(config['batch_size']):
                batch=images[indices.cuda()].float()/255
                optimizer.zero_grad(set_to_none=True)
                loss=reconstruction_loss(model(batch),batch,config['loss'])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite reconstruction loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                optimizer.step();total+=loss.item()*len(batch)
            torch.cuda.synchronize()
            train_seconds=time.perf_counter()-tick
            predicted=reconstruct(model,dev,config['batch_size'])
            dev_loss=reconstruction_loss(predicted,dev.cpu().float()/255,config['loss']).item()
            row=dict(epoch=epoch,train_loss=total/len(images),normal_development_loss=dev_loss,train_seconds=train_seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            history.append(row)
            pd.DataFrame(history).to_csv(compact/(identifier+'-history.csv'),index=False)
            if row['peak_allocated_bytes']>6*2**30 or time.perf_counter()-started>1200:
                raise RuntimeError('Per-run memory/time budget exceeded')
        torch.save(dict(state_dict=model.state_dict(),config=config,category=category,training_phase='final' if final else 'development'),checkpoint)
        torch.save(dict(optimizer=optimizer.state_dict(),shuffle_rng=shuffle.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),checkpoint.parent/'training-state.pt')
        metrics={}
        if not final:
            with torch.inference_mode():
                recon=torch.cat([model(b).cpu() for b in probes.split(config['batch_size'])])
            for kind in ['absolute','squared']:
                maps=anomaly_map(probes.cpu(),recon,kind)
                normal_maps=anomaly_map(dev.cpu().float()/255,predicted,kind)
                scores=image_scores(maps).numpy()
                labels=masks.cpu().flatten(1).any(1).numpy().astype(int)
                values=binary_metrics(labels,scores,calibrate_normal(image_scores(normal_maps).numpy()))
                values['pixel_ap']=float(average_precision_score(masks.cpu().numpy().ravel(),maps.numpy().ravel()))
                metrics[kind]=values
            torch.save(dict(original=dev[:3].cpu(),reconstruction=predicted[:3]),local/'development-reconstructions.pt')
        result=dict(experiment_id=identifier,category=category,configuration=config,configuration_sha256=config_hash(config),
            gradient_normal_images=len(learning),probe_normal_sources=0 if final else len(probe_rows),
            parameters=sum(p.numel() for p in model.parameters()),metrics=metrics,history=history,
            training_seconds_including_loading_validation=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha256(checkpoint),checkpoint_bytes=checkpoint.stat().st_size)
        with (compact/(identifier+'-result.json')).open('x') as stream:json.dump(result,stream,indent=2)
        event=copy.deepcopy(entry)
        event.update(experiment_id=identifier+'-result',status='completed',metrics=result,decision='await development comparison' if not final else 'frozen fit complete')
        record(ROOT/'experiments',event)
        print(identifier, 'completed',round(result['training_seconds_including_loading_validation'],2),metrics,flush=True)
        return result
    except BaseException as exc:
        (local/'failure.txt').write_text(traceback.format_exc())
        event=copy.deepcopy(entry)
        event.update(experiment_id=identifier+'-failure',status='failed',interpretation=str(exc),decision='preserve failure')
        record(ROOT/'experiments',event)
        raise
