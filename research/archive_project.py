"""Create a portable research archive, separate from the two submission ZIPs."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]

def include(path):
    rel=path.relative_to(ROOT)
    if any(part in ('__pycache__','.git','tmp') for part in rel.parts):return False
    if path.suffix in ('.pyc','.pkl'):return False
    if rel.parts[:2] in [('research','agents'),('research','stress_copies'),('research','historical_bundle')]:return False
    if len(rel.parts)>1 and rel.parts[0]=='research' and rel.parts[1].startswith('audit_agents_exact_'):return False
    return path.is_file()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--deliverables',type=Path,required=True)
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    files=[(f,'RMB_bateria_auditada/'+str(f.relative_to(ROOT))) for f in sorted(ROOT.rglob('*')) if include(f)]
    for name in ('submission_hybrid16_fulltrain.zip','agent_validat_prefinal.zip','informe_Riviera_Maya_Boys.pdf','delivery_verification.json'):
        f=a.deliverables/name
        files.append((f,'RMB_bateria_auditada/lliuraments/'+name))
    hashes={name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p,name in files}
    with ZipFile(a.out,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for path,name in files:z.write(path,name)
        z.writestr('RMB_bateria_auditada/ARCHIVE_HASHES.json',json.dumps(hashes,indent=2))
    with ZipFile(a.out) as z:
        bad=z.testzip();assert bad is None,bad
        assert 'RMB_bateria_auditada/README_REPRODUCCIO.md' in z.namelist()
    print(json.dumps(dict(file=str(a.out),bytes=a.out.stat().st_size,files=len(files),sha256=hashlib.sha256(a.out.read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':main()
