"""Public portfolio contracts; no model fitting, real images or private assets."""
from pathlib import Path
import copy
import pytest
import yaml
from scripts.check_portfolio import check,check_links,check_workflow

ROOT=Path(__file__).resolve().parents[1]


def test_public_documentation_metrics_and_schemas():
    report=check(ROOT)
    assert report['canonical_records']==270 and report['saved_code_cells']==205


def test_links_detect_missing_file_and_heading(tmp_path):
    page=tmp_path/'README.md';page.write_text('# Hello\n[missing](absent.md)\n')
    with pytest.raises(AssertionError):check_links(tmp_path,['README.md'])
    page.write_text('# Hello\n[missing](#absent)\n')
    with pytest.raises(AssertionError):check_links(tmp_path,['README.md'])
    page.write_text('# Hello\n[valid](#hello)\n')
    assert check_links(tmp_path,['README.md'])==1


def test_workflow_rejects_write_permissions_and_mutable_action_refs():
    workflow=yaml.load((ROOT/'.github/workflows/source-quality.yml').read_text(),Loader=yaml.BaseLoader)
    changed=copy.deepcopy(workflow);changed['permissions']['contents']='write'
    with pytest.raises(AssertionError):check_workflow(changed)
    changed=copy.deepcopy(workflow);changed['jobs']['frontend']['steps'][0]['uses']='actions/checkout@main'
    with pytest.raises(AssertionError):check_workflow(changed)
