import hashlib

import pytest
from pydantic import ValidationError

from loom.experiments import Trainer
from loom.trial_runner import artifact_digest


def test_realistic_checkpoint_above_legacy_limit(tmp_path):
    path = tmp_path / 'model.pth'
    with path.open('wb') as f:
        f.truncate(12_328_092)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert artifact_digest(tmp_path, 'model.pth', 256 * 1024**2) == expected
    with pytest.raises(ValueError, match='12328092 > 10485760'):
        artifact_digest(tmp_path, 'model.pth', 10 * 1024**2)


def test_artifact_limits_keep_path_and_size_guards(tmp_path):
    root = tmp_path / 'run'
    root.mkdir()
    (tmp_path / 'outside').write_bytes(b'123')
    (root / 'within').write_bytes(b'123')
    assert artifact_digest(root, 'within', 3)
    for name, limit in [('within', 2), ('../outside', 3), ('missing', 3), ('.', 3)]:
        with pytest.raises(ValueError):
            artifact_digest(root, name, limit)


def test_trainer_artifact_allowance_is_bounded():
    spec = dict(id='test', name='test', root='.', entrypoint='train.py', source_files=['train.py'],
                parameters={'x':[1]}, metrics=['loss'])
    assert Trainer(**spec).artifact_max_bytes == 256 * 1024**2
    for limit in (0, 2 * 1024**3 + 1):
        with pytest.raises(ValidationError):
            Trainer(**spec, artifact_max_bytes=limit)
