"""Pin Linux CUDA wheel extras from official PyTorch metadata (no wheel download)."""
from pathlib import Path
from html.parser import HTMLParser
from email import message_from_string
import json
import httpx
from packaging.requirements import Requirement
from packaging.markers import default_environment

ROOT=Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.links.append(dict(attrs).get('href',''))


def main():
    with httpx.Client(timeout=60,follow_redirects=True) as client:
        page=client.get('https://download.pytorch.org/whl/cu128/torch/');page.raise_for_status()
        parser=Links();parser.feed(page.text)
        urls=[u.split('#')[0] for u in parser.links if '2.11.0' in u and 'cp312-cp312-manylinux' in u and 'x86_64' in u]
        assert len(urls)==1,urls
        url=urls[0]+'.metadata'
        response=client.get(url);response.raise_for_status()
        metadata=message_from_string(response.text)
        pins=[]
        core={line.split('==')[0] for line in (ROOT/'deployment/requirements-runtime.txt').read_text().splitlines() if line and not line.startswith('#')}
        environment=dict(default_environment(),sys_platform='linux',platform_system='Linux',os_name='posix',extra='')
        for line in metadata.get_all('Requires-Dist',[]):
            item=Requirement(line)
            if item.name.lower().replace('_','-') not in core and (not item.marker or item.marker.evaluate(environment)):
                if item.name=='cuda-bindings':
                    # Official range <13,>=12.9.4; choose its documented lower bound.
                    item=Requirement('cuda-bindings==12.9.4')
                assert str(item.specifier).startswith('=='),line
                extras='['+','.join(sorted(item.extras))+']' if item.extras else ''
                pins.append(item.name+extras+str(item.specifier))
        assert pins
    # Binding loader dependency, pinned independently rather than floating at install.
    pins.append('cuda-pathfinder==1.4.0')
    (ROOT/'deployment/requirements-cuda-runtime.txt').write_text('# Official torch 2.11.0 cu128 Linux extras; separate from portable CPU image.\n'+'\n'.join(sorted(pins))+'\n',encoding='utf-8',newline='\n')
    (ROOT/'reports/deployment/cuda_metadata.json').write_text(json.dumps(dict(source=url,pins=pins),indent=2)+'\n',encoding='utf-8')
    print('Pinned',len(pins),'official Linux CUDA dependencies without downloading a pretraining dataset or CUDA wheel')


if __name__=='__main__':main()
