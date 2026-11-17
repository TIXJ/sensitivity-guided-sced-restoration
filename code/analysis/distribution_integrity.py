"""Validate distributed bytes before materializing physical experiments."""
from pathlib import Path
import hashlib
import json
import os


def write_only(mode, flags):
    if isinstance(mode, str):
        return '+' not in mode and any(character in mode for character in 'wax')
    access = os.O_RDONLY | os.O_WRONLY | os.O_RDWR
    return flags & access == os.O_WRONLY


def allowed_read(path, roots):
    path = Path(path).resolve()
    return path == Path(os.devnull).resolve() or any(path.is_relative_to(Path(root).resolve()) for root in roots)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(repository):
    repository = Path(repository).resolve()
    manifest = repository / 'MANIFEST.json'
    files = json.loads(manifest.read_text(encoding='utf-8'))['files']
    errors = []
    for name, expected in files.items():
        path = (repository / name).resolve()
        if not path.is_relative_to(repository) or not path.is_file():
            errors.append(name + ': missing or outside repository')
        elif digest(path) != expected:
            errors.append(name + ': SHA256 mismatch')
    if errors:
        raise ValueError('Distribution verification failed:\n' + '\n'.join(errors))
    return {'manifest_sha256': digest(manifest), 'verified_files': len(files)}
