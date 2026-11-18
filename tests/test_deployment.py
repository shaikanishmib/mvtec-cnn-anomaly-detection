"""Deployment-only contracts with authored artifact bytes; no research refitting."""
import json
from pathlib import Path
import pytest
from deployment import artifacts, runtime
from anomaly_detection.inference import ArtifactError
from anomaly_detection.tracking import config_hash


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root=tmp_path/'bundle';root.mkdir()
    def entry(path,data):
        target=root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        return dict(path=path,bytes=len(data),sha256=artifacts.sha256(target))
    config={'bank_budget':2048,'seed':2026}
    manifest=dict(seed=2026,preprocessing={'version':'authored-unit-fixture'},configuration=config,categories={})
    manifest['preprocessing_sha256']=config_hash(manifest['preprocessing'])
    manifest['frozen_policy']=entry('configs/policy.json',json.dumps(dict(configuration=config)).encode())
    manifest['extractor']=entry('checkpoints/extractor.pt',b'authored dummy; never loaded as a model')
    for category in artifacts.CATEGORIES:
        manifest['categories'][category]=dict(bank=entry(f'checkpoints/{category}/bank.pt',category.encode()),
            calibration=entry(f'calibration/{category}.json',b'{"image_threshold": 1.5, "pixel_threshold": 0.4}'),
            thresholds=dict(image=1.5,pixel=.4,strict_greater=True))
    def save():
        path=root/artifacts.MANIFEST;path.write_text(json.dumps(manifest),encoding='utf-8')
        monkeypatch.setattr(artifacts,'PIN',artifacts.sha256(path))
    save()
    return root,manifest,save


def test_all_category_artifact_validation(bundle):
    root,_,_=bundle
    _,entries=artifacts.validate(root)
    assert len(entries)==32


@pytest.mark.parametrize('mode',['missing','corrupt','wrong_size'])
def test_bad_bank(bundle,mode):
    root,m,save=bundle;entry=m['categories']['bottle']['bank'];path=root/entry['path']
    if mode=='missing':path.unlink()
    elif mode=='corrupt':path.write_bytes(b'x'*entry['bytes'])
    else:entry['bytes']+=1;save()
    with pytest.raises(ArtifactError):artifacts.validate(root)


@pytest.mark.parametrize('mode',['seed','categories','preprocessing','threshold','configuration','escape'])
def test_manifest_policy_integrity(bundle,mode):
    root,m,save=bundle
    if mode=='seed':m['seed']=2028
    if mode=='categories':del m['categories']['bottle']
    if mode=='preprocessing':m['preprocessing']['version']='changed'
    if mode=='threshold':m['categories']['bottle']['thresholds']['image']=2.
    if mode=='configuration':m['configuration']['bank_budget']=1
    if mode=='escape':m['extractor']['path']='../outside.pt'
    save()
    with pytest.raises(ArtifactError):artifacts.validate(root)


def test_manifest_checksum(bundle):
    root,_,_=bundle;(root/artifacts.MANIFEST).write_bytes(b'{}')
    with pytest.raises(ArtifactError):artifacts.validate(root)


def test_export_is_minimal_repeatable_and_preserves_existing(bundle,tmp_path):
    root,_,_=bundle;destination=tmp_path/'export'
    report=artifacts.export(root,destination);assert report['files']==33
    assert len([p for p in destination.rglob('*') if p.is_file()])==33
    assert artifacts.export(root,destination)==report
    original=(destination/'checkpoints/bottle/bank.pt');original.write_bytes(b'broken')
    with pytest.raises(ArtifactError):artifacts.export(root,destination)
    assert original.read_bytes()==b'broken'


def test_export_cannot_target_source(bundle):
    root,_,_=bundle
    with pytest.raises(ArtifactError):artifacts.export(root,root)


def test_entrypoint_cuda_failure_never_falls_back(monkeypatch):
    import torch
    monkeypatch.setattr(runtime,'validate',lambda _:None)
    monkeypatch.setenv('ANOMALY_DEVICE','cuda')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(SystemExit) as exc:runtime.main()
    assert exc.value.code==1


def test_entrypoint_exec_one_worker(monkeypatch):
    monkeypatch.setattr(runtime,'validate',lambda _:None)
    monkeypatch.setenv('ANOMALY_DEVICE','cpu')
    class Executed(Exception):pass
    def execute(binary,args):
        assert args[args.index('--workers')+1]=='1'
        assert args[-1]=='--no-access-log'
        raise Executed
    monkeypatch.setattr(runtime.os,'execv',execute)
    with pytest.raises(Executed):runtime.main()


def test_container_build_context_and_nonroot_contract():
    root=Path(__file__).resolve().parents[1]
    ignore=(root/'.dockerignore').read_text()
    assert '\n**\n' in ignore and '!frontend/src/**' in ignore
    assert '!checkpoints' not in ignore and '!data/' not in ignore
    backend=(root/'deployment/Dockerfile.backend').read_text()
    assert 'USER 10001:10001' in backend and '@sha256:' in backend
    assert 'COPY checkpoints' not in backend and 'COPY data' not in backend
    frontend=(root/'deployment/Dockerfile.frontend').read_text()
    assert 'USER 101:101' in frontend and frontend.count('@sha256:')==2
    compose=(root/'compose.yaml').read_text()
    assert '127.0.0.1:' in compose and 'read_only: true' in compose


def test_build_recovery_with_published_evidence_but_no_local_image(tmp_path,monkeypatch):
    import subprocess
    import scripts.release_deployment as release
    out=tmp_path/'reports';out.mkdir();local=tmp_path/'outputs'
    (out/'builds.json').write_text(json.dumps(dict(backend=dict(source_sha256='same',image_id='old'))))
    monkeypatch.setattr(release,'OUT',out);monkeypatch.setattr(release,'LOCAL',local)
    monkeypatch.setattr(release,'source_hash',lambda *args:'same')
    calls=[]
    def inspect(*args):
        calls.append(args)
        if len(calls)==1:raise subprocess.CalledProcessError(1,['docker','image','inspect'])
        return [dict(Id='new',Size=123,Config=dict(User='10001:10001'),Os='linux',Architecture='amd64')]
    monkeypatch.setattr(release,'docker_json',inspect)
    class Success:returncode=0
    monkeypatch.setattr(release.subprocess,'run',lambda *args,**kwargs:Success())
    release.build('backend')
    assert json.loads((out/'builds.json').read_text())['backend']['image_id']=='new'
