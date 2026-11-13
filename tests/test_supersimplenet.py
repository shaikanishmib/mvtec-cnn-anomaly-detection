import io
import numpy as np
import pytest
import torch
from anomaly_detection.supersimplenet import (SSNExtractor,SuperSimpleHeads,feature_anomalies,training_loss,
    anomaly_maps,image_scores,perlin_noise)


def test_alignment_and_deterministic_training_step():
    from anomaly_detection.supersimplenet import aggregate_features,optimizer_for
    a=torch.ones(2,3,8,8);b=torch.ones(2,5,4,4)*2
    aligned=aggregate_features(a,b)
    assert aligned.shape==(2,8,16,16)
    assert torch.equal(aligned[:,0,2:-2,2:-2],torch.ones(2,12,12))
    def run():
        torch.manual_seed(401)
        model=SuperSimpleHeads(8,4,2);optimizer=optimizer_for(model)
        features=torch.randn(3,8,16,16)
        outputs=model.synthetic_forward(features,torch.Generator().manual_seed(5),torch.Generator().manual_seed(6))
        loss,_=training_loss(*outputs);loss.backward();optimizer.step()
        return {key:value.detach().clone() for key,value in model.state_dict().items()}
    first,second=run(),run()
    assert all(torch.equal(first[key],second[key]) for key in first)


def test_checkpoint_category_and_configuration_isolation(tmp_path):
    from anomaly_detection.ssn_evaluation import load_heads
    model=SuperSimpleHeads(8,4,2);path=tmp_path/'heads.pt'
    torch.save(dict(category='bottle',configuration={'seed':2026},state_dict=model.state_dict()),path)
    with pytest.raises(ValueError,match='isolation'):load_heads(path,'cable',{'seed':2026})
    with pytest.raises(ValueError,match='isolation'):load_heads(path,'bottle',{'seed':2027})


def test_calibration_rejects_nonreserved_or_anomalous_rows(monkeypatch):
    import pandas as pd
    from anomaly_detection import ssn_evaluation as evaluation
    for partition,mask in [('fit',None),('test','defect.png'),('calibration','defect.png')]:
        monkeypatch.setattr(evaluation,'read_partition',lambda *args,p=partition,m=mask:pd.DataFrame(dict(partition=[p],mask=[m])))
        with pytest.raises(ValueError,match='reserved'):evaluation.calibrate_model(None,'bottle',256)


def test_shared_metric_compatibility_and_identity():
    from anomaly_detection.metrics import binary_metrics,calibrate_normal
    from anomaly_detection.tracking import config_hash
    config=dict(model='SuperSimpleNetLite',seed=2026,noise_std=.015)
    assert config_hash(config)==config_hash(dict(reversed(list(config.items()))))
    assert config_hash(config)!=config_hash(dict(config,seed=2027))
    threshold=calibrate_normal([-4,-3,-2,-1])
    value=binary_metrics([0,0,1,1],[-3,-2,1,2],threshold)
    assert value['auroc']==value['ap']==value['recall']==value['f1']==1
    assert value['fpr']==0 and value['confusion_matrix']==[[2,0],[0,2]]


def test_clean_inference_and_reserved_normal_calibration(monkeypatch):
    import pandas as pd
    from types import SimpleNamespace
    from anomaly_detection import ssn_evaluation as evaluation
    from anomaly_detection.ssn_research import predict
    from anomaly_detection.metrics import calibrate_normal
    x=torch.randn(3,8,16,16);original=x.clone()
    class Features:
        metadata={'input_rows_sha256':'fixture'}
        def __len__(self):return len(x)
        def batch(self,indices):return x[list(indices)]
    features=Features();heads=SuperSimpleHeads(8,4,2)
    def forbidden(*args,**kwargs):raise AssertionError('Synthetic perturbations entered inference/calibration')
    monkeypatch.setattr(heads,'synthetic_forward',forbidden)
    maps,scores,_=predict(heads,features,batch_size=2)
    assert maps.shape==(3,256,256) and torch.isfinite(scores).all()
    assert torch.equal(x,original) and not heads.training
    rows=pd.DataFrame(dict(partition=['calibration']*3,mask=[None]*3,path=['a','b','c'],sha256=['1','2','3']))
    monkeypatch.setattr(evaluation,'read_partition',lambda *args:rows)
    monkeypatch.setattr(evaluation,'feature_cache',lambda *args:features)
    model=SimpleNamespace(extractor=None,heads=heads,config=dict(extraction_batch=8,batch_size=2,aggregation='head'))
    value=evaluation.calibrate_model(model,'bottle',256)
    assert value['images']==3 and value['pixel_calibration_samples']==3*256*256
    assert value['image_threshold']==calibrate_normal(scores.numpy())
    assert value['pixel_threshold']==float(__import__('numpy').quantile(maps.numpy(),.995,method='higher'))
    assert torch.equal(x,original)


def test_normal_only_fitting_and_single_category_cache_guard():
    import pandas as pd
    from anomaly_detection.cnn_research import require_fitting
    from anomaly_detection.ssn_research import feature_cache
    normal=pd.DataFrame(dict(partition=['fit'],split=['train'],mask=[None]))
    require_fitting(normal)
    for changed in [normal.assign(partition='development'),normal.assign(split='test'),normal.assign(mask='mask.png')]:
        with pytest.raises(ValueError,match='fitting normals'):require_fitting(changed)
    with pytest.raises(ValueError,match='Single nonempty category'):
        feature_cache(None,pd.DataFrame(dict(category=['bottle','cable'])))


def test_durable_resume_matches_uninterrupted_optimization(tmp_path,monkeypatch):
    from anomaly_detection import ssn_final as final
    from anomaly_detection.ssn_research import train_epoch
    from anomaly_detection.supersimplenet import optimizer_for
    monkeypatch.setattr(final,'ROOT',tmp_path)
    path=tmp_path/'checkpoints/supersimplenet/fixture/progress.pt';path.parent.mkdir(parents=True)
    config=dict(batch_size=2,noise_std=.05,loss='margin_focal')
    identity=dict(category='bottle',configuration=config,source='fixture')
    x=torch.randn(4,8,16,16,generator=torch.Generator().manual_seed(7))
    class Features:
        def __len__(self):return len(x)
        def batch(self,indices):return x[indices]
    def setup():
        torch.manual_seed(401);model=SuperSimpleHeads(8,4,2);optimizer=optimizer_for(model)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[2],gamma=.4)
        generators={key:torch.Generator().manual_seed(seed) for key,seed in [('shuffle',1),('mask',2),('noise',3)]}
        return model,optimizer,scheduler,generators
    model,optimizer,scheduler,generators=setup();history=[]
    def epoch(model,optimizer,scheduler,generators):
        value=train_epoch(model,Features(),optimizer,generators['shuffle'],generators['mask'],generators['noise'],config)
        scheduler.step();return value
    for _ in range(2):history.append(epoch(model,optimizer,scheduler,generators))
    final.save_progress(path,model,optimizer,scheduler,generators,history,identity,{})
    expected_random=torch.rand(3);expected=epoch(model,optimizer,scheduler,generators)
    resumed,opt,sch,gen=setup();saved=final.restore_progress(path,identity,resumed,opt,sch,gen)
    assert len(saved['history'])==2 and torch.equal(torch.rand(3),expected_random)
    actual=epoch(resumed,opt,sch,gen)
    assert actual==expected
    assert all(torch.equal(value,resumed.state_dict()[key]) for key,value in model.state_dict().items())
    with pytest.raises(ValueError,match='identity mismatch'):
        final.restore_progress(path,dict(identity,category='cable'),resumed,opt,sch,gen)
    path.with_suffix('.pending.pt').write_bytes(b'interrupted partial write')
    final.save_progress(path,resumed,opt,sch,gen,saved['history'],identity,{})
    assert len(list(path.parent.glob('interrupted-write-*.pt')))==1


def test_extractor_shapes_and_frozen_statistics():
    torch.set_num_threads(4)
    model=SSNExtractor(pretrained=False);model.train()
    x=torch.rand(1,3,256,256)
    a,b=model.feature_maps(x)
    assert a.shape==(1,128,32,32) and b.shape==(1,256,16,16)
    assert model(x).shape==(1,384,64,64)
    assert not model.training and all(not p.requires_grad for p in model.parameters())


def test_perlin_and_exact_noise_support():
    a=perlin_noise(16,16,4,2,torch.Generator().manual_seed(1))
    b=perlin_noise(16,16,4,2,torch.Generator().manual_seed(1))
    assert torch.equal(a,b) and a.std()>0
    x=torch.randn(4,8,16,16);original=x.clone()
    changed,mask,labels=feature_anomalies(x,torch.Generator().manual_seed(3),torch.Generator().manual_seed(4),threshold=-.2,normal_probability=0)
    actual=(changed-torch.cat([x,x])).ne(0).any(1,keepdim=True)
    assert torch.equal(actual,mask.bool()) and torch.equal(x,original)
    assert torch.equal(labels,mask.flatten(1).any(1).float())
    clean,m,y=feature_anomalies(x,torch.Generator().manual_seed(3),torch.Generator().manual_seed(4),normal_probability=1)
    assert torch.equal(clean,torch.cat([x,x])) and not m.any() and not y.any()


def test_outputs_losses_gradients_and_checkpoint():
    model=SuperSimpleHeads(8,16,4)
    x=torch.rand(3,8,16,16)
    seg,score,mask,label=model.synthetic_forward(x,torch.Generator().manual_seed(1),torch.Generator().manual_seed(2))
    assert seg.shape==(6,1,16,16) and score.shape==(6,)
    loss,parts=training_loss(seg,score,mask,label)
    assert torch.isfinite(loss);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval();first=model(x)
    stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
    restored=SuperSimpleHeads(8,16,4).eval();restored.load_state_dict(torch.load(stream,weights_only=True))
    assert all(torch.equal(a,b) for a,b in zip(first,restored(x)))
    assert all(torch.equal(a,b) for a,b in zip(model(x),model(x)))


def test_classification_stop_gradient_and_fixed_adapter():
    model=SuperSimpleHeads(8,16,4,stop_grad=True)
    _,score=model(torch.randn(3,8,16,16));score.sum().backward()
    assert all(p.grad is None for p in model.seg.parameters())
    assert model.adapter.weight.grad is not None
    fixed=SuperSimpleHeads(8,16,4,adapter_trainable=False)
    s,d=fixed(torch.randn(3,8,16,16));(s.mean()+d.mean()).backward()
    assert all(p.grad is None for p in fixed.adapter.parameters())


def test_orientation_maps_aggregation_and_metric_compatibility():
    from anomaly_detection.metrics import calibrate_normal,binary_metrics
    target=torch.tensor([[[[0.,1.],[0.,1.]]]])
    correct=4*(target-.5)
    a,_=training_loss(correct,torch.tensor([2.]),target,torch.ones(1))
    b,_=training_loss(-correct,torch.tensor([-2.]),target,torch.ones(1))
    assert a<b
    maps=anomaly_maps(torch.ones(2,1,16,16)*-2)
    assert maps.shape==(2,256,256) and torch.allclose(maps,torch.ones_like(maps)*-2,atol=1e-6)
    assert image_scores(maps,torch.tensor([1.,3.])).tolist()==[1,3]
    assert image_scores(maps,torch.tensor([1.,3.]),'max').tolist()==pytest.approx([-2,-2],abs=1e-6)
    threshold=calibrate_normal(np.arange(20))
    assert binary_metrics([0,0,1,1],[18,19,20,21],threshold)['confusion_matrix']==[[2,0],[0,2]]
