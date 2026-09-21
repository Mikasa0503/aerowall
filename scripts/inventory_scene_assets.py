"""Inventory authored USD dependencies without fetching remote resources."""
import argparse,hashlib,json
from pathlib import Path
from pxr import UsdUtils
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
roots=[ROOT/'third_party/JuggleRL_train/omni_drones/envs/assets/default_environment.usd',ROOT/'third_party/JuggleRL_train/omni_drones/robots/assets/usd/air.usd']
mdl=ROOT/'third_party/isaac-sim-2023.1.0-hotfix.1/kit/mdl/core/Base'
queue=list(roots);seen=set();files=[];links=[]
while queue:
 path=queue.pop().resolve()
 if path in seen:continue
 seen.add(path);assert path.is_relative_to(ROOT),path
 if not path.is_file():files.append({'path':str(path.relative_to(ROOT)),'exists':False});continue
 files.append({'path':str(path.relative_to(ROOT)),'exists':True,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 if path.suffix.lower() not in ['.usd','.usda','.usdc']:continue
 for category,items in zip(['sublayer','reference_or_asset','payload'],UsdUtils.ExtractExternalReferences(str(path))):
  for item in items:
   if '://' in item:
    links.append({'from':str(path.relative_to(ROOT)),'authored':item,'category':category,'remote':True});continue
   dependency=mdl/item if item=='OmniPBR.mdl' else path.parent/item
   links.append({'from':str(path.relative_to(ROOT)),'authored':item,'category':category,'remote':False,'resolved_candidate':str(dependency.resolve().relative_to(ROOT))})
   queue.append(dependency)
result={'scope':'Static authored USD dependency inventory for the configured local ground and Air model. Core MDL candidate recorded; does not validate runtime resolver choice, transitive MDL imports, extension network access or complete offline operation.',
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'files':files,'links':links,
 'missing_local_files':sum(not f['exists'] for f in files),'remote_usd_references':sum(e['remote'] for e in links)}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['files','links']}))
