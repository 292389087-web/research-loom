"""Bounded dependency checks for locally registered, trusted training code."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from .config import ROOT


def asset_snapshot(adapter):
    result={}
    for name, location in adapter.get('required_assets',{}).items():
        path=Path(location).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f'模型/资源文件缺失：{name}: {path}')
        with path.open('rb') as stream:
            result[name]={'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.file_digest(stream,'sha256').hexdigest()}
    return result


def check_preflight(adapter, manifest=None, timeout=30):
    from .experiments import resolve_file, file_digest, dataset_snapshots
    errors=[]; modules=set(adapter.get('required_modules',[])); sources={}; assets={}
    root=Path(adapter['root']).resolve()
    try:
        for name in [*adapter['source_files'],*adapter.get('dataset_files',[])]:
            path=resolve_file(root,name);sources[name]=file_digest(path)
            if name in adapter['source_files'] and path.suffix=='.py':
                tree=ast.parse(path.read_text(encoding='utf-8-sig'),filename=name)
                # Mandatory top-level imports; conditional/dynamic imports
                # must be explicitly declared in required_modules.
                for node in tree.body:
                    if isinstance(node,ast.Import):modules.update(n.name for n in node.names)
                    elif isinstance(node,ast.ImportFrom) and not node.level and node.module:modules.add(node.module)
        assets=asset_snapshot(adapter)
        datasets=dataset_snapshots(adapter)
        if manifest:
            expected={k:v for k,v in manifest['source_hashes'].items() if k in [*adapter['source_files'],*adapter.get('dataset_files',[])]}
            if sources!=expected:errors.append('源码与冻结计划不一致，请新建计划')
            if datasets!=manifest.get('datasets',[]):errors.append('数据与冻结计划不一致')
            if assets!=manifest.get('assets',{}):errors.append('模型/资源与冻结计划不一致')
    except (ValueError,OSError,SyntaxError) as exc:
        errors.append(str(exc))
    imports=[]
    if not errors:
        # Import dependencies, never execute/import the training entrypoint.
        program='''import importlib,json,sys
sys.path[:0]=json.loads(sys.argv[1])
for name in json.loads(sys.argv[2]):
 try: importlib.import_module(name)
 except Exception as e: print("LOOM_PREFLIGHT_ERROR:"+name+": "+type(e).__name__+": "+str(e)[:500])
'''
        entry=Path(adapter['entrypoint']).with_suffix('').as_posix().replace('/','.')
        imports=sorted(modules-{entry,'__future__'})
        try:
            result=subprocess.run([sys.executable,'-X','utf8','-c',program,json.dumps([str(root),str(ROOT)]),json.dumps(imports)],
                cwd=root,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=timeout)
            errors.extend(line.split('LOOM_PREFLIGHT_ERROR:',1)[1] for line in result.stdout.splitlines() if line.startswith('LOOM_PREFLIGHT_ERROR:'))
            if result.returncode:errors.append('依赖检查进程异常退出：'+result.stderr[-1500:])
        except subprocess.TimeoutExpired:errors.append(f'依赖导入检查超过 {timeout} 秒')
        except OSError as exc:errors.append(str(exc))
    return {'passed':not errors,'errors':errors,'imports':imports,'source_hashes':sources,'assets':assets,
            'scope':'静态入口检查、顶层依赖导入和已声明数据/资产；不执行训练，不保证动态分支或算法正确。'}
