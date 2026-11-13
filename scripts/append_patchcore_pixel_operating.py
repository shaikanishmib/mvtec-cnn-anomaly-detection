"""Execute a compact supplemental report without replaying feature extraction."""
from pathlib import Path
import os,asyncio
import nbformat as nbf
from nbclient import NotebookClient
root=Path(__file__).resolve().parents[1]
# Assemble the publisher's cells without executing its full analysis again.
p=root/'scripts/publish_patchcore_notebook.py';scope={'__file__':str(p)}
exec(compile(p.read_text(encoding='utf-8').split('NotebookClient(tail,',1)[0],str(p),'exec'),scope)
extra=nbf.v4.new_notebook()
extra.metadata.kernelspec=scope['tail'].metadata.kernelspec
extra.cells=[c for c in scope['tail'].cells if c.metadata.get('pixel_operating_analysis')]
assert len(extra.cells)==2
if os.name=='nt':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
NotebookClient(extra,timeout=120,kernel_name='mvtec-research',resources={'metadata':{'path':str(root)}}).execute()
for path in [root/'notebooks/03_patchcore.ipynb',root/'outputs/patchcore/03_patchcore.complete.executed.ipynb']:
    notebook=nbf.read(path,as_version=4)
    notebook.cells=[c for c in notebook.cells if not c.metadata.get('pixel_operating_analysis')]
    index=next((i for i,c in enumerate(notebook.cells) if c.metadata.get('generated_patchcore_findings')),len(notebook.cells)-1)
    notebook.cells[index:index]=extra.cells
    path.write_bytes(nbf.writes(notebook).encode('utf-8'))
print('Executed and retained supplemental pixel operating-point analysis; no feature extraction/scoring repeated')
