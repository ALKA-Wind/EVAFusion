"""Reassemble the reward checkpoint from Git LFS parts, verifying SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path


def _digest(path):
    value=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(8*1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def _local_file(directory, name):
    if not isinstance(name,str) or not name or Path(name).name!=name or '/' in name or '\\' in name:
        raise ValueError('Checkpoint manifest filenames must be local basenames')
    return directory/name


def restore_checkpoint(directory=None):
    directory=Path(directory or Path(__file__).resolve().parent/'20250503_232931')
    manifest=json.loads((directory/'checkpoint_manifest.json').read_text(encoding='utf-8'))
    target=_local_file(directory,manifest['file'])
    if target.exists():
        if target.stat().st_size!=manifest['bytes'] or _digest(target)!=manifest['sha256']:
            raise ValueError(f'Existing checkpoint differs from manifest; refusing to overwrite: {target}')
        return target
    parts=[(_local_file(directory,item['file']),item) for item in manifest['parts']]
    if not parts or sum(item['bytes'] for _,item in parts)!=manifest['bytes']:
        raise ValueError('Invalid checkpoint manifest part sizes')
    for path,item in parts:
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}; run git lfs pull first')
        if path.stat().st_size!=item['bytes']:
            raise ValueError(f'Part size differs from manifest: {path}; run git lfs pull')
    temporary=target.with_name(target.name+'.assembling')
    combined=hashlib.sha256()
    created=False
    try:
        with temporary.open('xb') as output:
            created=True
            for path,item in parts:
                digest=hashlib.sha256()
                with path.open('rb') as part:
                    for block in iter(lambda:part.read(8*1024*1024),b''):
                        output.write(block); digest.update(block); combined.update(block)
                if digest.hexdigest()!=item['sha256']:
                    raise ValueError(f'Checkpoint part checksum mismatch: {path}')
        if combined.hexdigest()!=manifest['sha256'] or temporary.stat().st_size!=manifest['bytes']:
            raise ValueError('Combined checkpoint checksum mismatch')
        if target.exists():
            raise FileExistsError(f'Checkpoint appeared during restoration: {target}')
        temporary.rename(target)
        return target
    finally:
        if created and temporary.exists():
            temporary.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path)
    args=parser.parse_args()
    print(f'Verified reward checkpoint: {restore_checkpoint(args.directory).resolve()}')


if __name__=='__main__':
    main()
