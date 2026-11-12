"""Derive AE evaluator from verified native-metric implementation, preserving CNN."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=(root/'src/anomaly_detection/cnn_evaluation.py').read_text()
source=source.replace('from .basic_cnn import BasicCNN, image_scores','from .autoencoder import ConvAutoencoder, anomaly_map, image_scores')
source=source.replace('from .cnn_research import ROOT, read_partition, load_resized, predict_logits','from .cnn_research import ROOT, read_partition, load_resized\nfrom .ae_research import reconstruct')
source=source.replace('model = BasicCNN(saved["config"]["width"])','model = ConvAutoencoder(**{k:saved["config"][k] for k in ["width","depth","latent_channels"]})')
source=source.replace('return model.cuda().eval(), saved','model = model.cuda().eval()\n    model.map_kind = saved["config"]["map_kind"]\n    return model, saved')
source=source.replace('    logits = predict_logits(model, images)\n    scores = image_scores(logits).numpy()',
    '    maps = anomaly_map(images.cpu().float()/255, reconstruct(model, images), model.map_kind)\n    scores = image_scores(maps).numpy()')
source=source.replace('probabilities = logits.sigmoid().numpy().ravel()','probabilities = maps.numpy().ravel()')
source=source.replace('logits = model(x); torch.cuda.synchronize()', 'reconstruction = model(x)\n        maps = anomaly_map(x, reconstruction, model.map_kind).unsqueeze(1); torch.cuda.synchronize()')
source=source.replace('F.interpolate(logits.sigmoid(), size=native_shape','F.interpolate(maps, size=native_shape')
source=source.replace('float(result.max())','float(image_scores(maps[:,0]).item())')
source=source.replace('native probability map','native residual map')
source=source.replace('basic_cnn_frozen.json','autoencoder_frozen.json').replace('reports/basic_cnn','reports/autoencoder').replace('outputs/basic_cnn','outputs/autoencoder')
source=source.replace('"model_family":"BasicCNN"','"model_family":"ConvAutoencoder"')
source=source.replace('Frozen synthetic-supervised CNN transfers to official real defects; no further selection','Frozen normal-only reconstruction baseline; no further selection')
source=source.replace('"Frozen real-test confirmation; no post-test changes permitted"','"Frozen real-test confirmation; no post-test changes permitted"')
source=source.replace('        logits=predict_logits(model,images,config["batch_size"])\n        maps=logits.sigmoid().numpy()[:,0]\n        # Primary maximum score at model resolution, declared before test access.\n        scores=maps.reshape(len(maps),-1).max(1)',
    '        maps=anomaly_map(images.cpu().float()/255,reconstruct(model,images,config["batch_size"]),model.map_kind).numpy()\n        # Frozen top-1% mean at model resolution.\n        scores=image_scores(torch.from_numpy(maps)).numpy()')
# Keep shared native metrics/bootstrap implementation, with explicit residual-bin labels.
source=source.replace('fixed probability bins','fixed [0,1] residual bins')
source=source.replace('from .cnn_research import code_identity','from .cnn_research import code_identity')
source=source.replace('    expected=dict(frozen["configuration"]',
    '    for path, expected_hash in frozen["source_sha256"].items():\n        if sha256(ROOT/path)!=expected_hash:raise ValueError("Frozen model/training source changed")\n    expected=dict(frozen["configuration"]')
source=source.replace('"code":code_identity(),',
    '"code":{"source_sha256":frozen["source_sha256"],"git_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()},')
(root/'src/anomaly_detection/ae_evaluation.py').write_text(source)
