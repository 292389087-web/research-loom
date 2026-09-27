import json
import subprocess
import sys
import pytest
from pathlib import Path


def command(root, *args):
    return subprocess.run(
        [sys.executable, "-X", "utf8", *map(str, args)],
        cwd=root,
        capture_output=True,
        encoding='utf-8',
        timeout=20,
    )


def test_build_and_verify_evidence_pack(tmp_path):
    root = tmp_path / "source"
    (root / "reports").mkdir(parents=True)
    (root / "reports" / "result.json").write_text('{"passed": true}', encoding="utf-8")
    spec = root / "spec.json"
    spec.write_text(json.dumps({"entries": [{"pattern": "reports/*.json"}]}), encoding="utf-8")
    output = tmp_path / "pack"
    project = Path(__file__).resolve().parents[1]
    build = command(project, project / "scripts" / "build_evidence_pack.py", "--root", root, "--spec", spec, "--output", output)
    assert build.returncode == 0, build.stdout + build.stderr
    verify = command(project, project / "scripts" / "verify_evidence_pack.py", output)
    assert verify.returncode == 0, verify.stdout + verify.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"][0]["path"] == "reports/result.json"


def test_evidence_pack_rejects_secret(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "report.txt").write_text("api_key = '" + "x" * 30 + "'", encoding="utf-8")
    spec = root / "spec.json"
    spec.write_text(json.dumps({"entries": [{"pattern": "report.txt"}]}), encoding="utf-8")
    project = Path(__file__).resolve().parents[1]
    result = command(project, project / "scripts" / "build_evidence_pack.py", "--root", root, "--spec", spec, "--output", tmp_path / "pack")
    assert result.returncode != 0
    assert "疑似包含密钥" in result.stderr


@pytest.mark.parametrize('name,content', [
    ('report.json', '{"api_key": "'+'x'*30+'"}'),
    ('report.json', '{"Authorization": "Bearer '+'x'*30+'"}'),
    ('other.local.toml', 'secret'), ('.env.production', 'secret'),
])
def test_pack_rejects_json_secrets_and_local_configs(tmp_path, name, content):
    from scripts.build_evidence_pack import collect
    (tmp_path/name).write_text(content, encoding='utf-8')
    with pytest.raises(ValueError):
        collect(tmp_path, {'entries':[{'pattern':name}]})


@pytest.mark.parametrize('attack', ['tamper', 'extra', 'duplicate', 'escape', 'empty'])
def test_verify_rejects_corrupted_or_incomplete_pack(tmp_path, attack):
    import hashlib
    pack = tmp_path/'pack'; pack.mkdir()
    data = b'original'; (pack/'result.txt').write_bytes(data)
    rec = {'path':'result.txt','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    manifest = {'schema':'research-loom-evidence-pack-v1','files':[rec]}
    if attack == 'tamper': (pack/'result.txt').write_bytes(b'changed!')
    if attack == 'extra': (pack/'unlisted.txt').write_text('extra')
    if attack == 'duplicate': manifest['files'].append(rec)
    if attack == 'escape': rec['path'] = '../outside.txt'
    if attack == 'empty': manifest['files'] = []
    (pack/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    root = Path(__file__).resolve().parents[1]
    result = command(root, root/'scripts/verify_evidence_pack.py', pack)
    assert result.returncode != 0
