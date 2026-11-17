"""After extracting the reproducibility ZIP, relocate legacy absolute root paths."""
from pathlib import Path
import json

def main():
    package=Path(__file__).resolve().parent
    settings_path=package/'settings.json'
    settings=json.loads(settings_path.read_text(encoding='utf-8'))
    previous=settings['packaged_project_root']
    current=(package/'ED_RL').resolve().as_posix()
    if previous!=current:
        for path in (package/'ED_RL').rglob('*.py'):
            text=path.read_text(encoding='utf-8-sig').replace(previous,current)
            path.write_text(text,encoding='utf-8')
        settings['packaged_project_root']=current
        settings_path.write_text(json.dumps(settings,indent=2),encoding='utf-8')
    print('Project root:',current)

if __name__=='__main__': main()
