"""Verify the public demo without importing GPU or training libraries."""
import os
from pathlib import Path
import subprocess
import sys


def test_cpu_demo_without_training_dependencies(tmp_path):
    script = r'''
import importlib.abc, sys
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','sentence_transformers','optuna','psutil'}:
            raise AssertionError('demo imported training dependency: '+fullname)
sys.meta_path.insert(0, BlockTraining())
from fastapi.testclient import TestClient
from loom.api import create_app
from loom.config import Settings
from pathlib import Path
app=create_app(Settings(data_dir=Path(sys.argv[1]),demo_mode=True,worker_enabled=False,
                        allowed_hosts=('testserver',)),start_worker=False)
with TestClient(app) as c:
    assert c.get('/workspace').status_code==200
    assert c.get('/api/models').json()['demo_mode'] is True
    ds=c.post('/api/data/demo').json()
    r=c.post('/api/data/datasets/'+ds['id']+'/analyze',json={'operation':'compare'})
    assert r.status_code==200, r.text
    assert r.json()['dataset_kind']=='synthetic_demo'
    assert c.post('/api/data/questions',json={'dataset_id':ds['id'],'question':'compare all metrics'}).status_code==422
    assert c.post('/api/studies/unused/start').status_code==422
print('CPU demo passed without GPU/training imports')
'''
    result=subprocess.run([sys.executable,'-X','utf8','-c',script,str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,encoding='utf-8',
        env={**os.environ,'LOOM_REMOTE_CONFIG':str(tmp_path/'missing.toml')},timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
