"""Offline experiment bundles; one writer, local filesystem, Python stdlib only.

Integrity checks establish byte identity, not controller efficacy. A process
interruption before pointer replacement preserves the previously selected bundle.
No power-loss or adversarial-write guarantees are claimed.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != 1 or not manifest.get("files"):
        raise ValueError("Invalid manifest")
    for name, expected in manifest["files"].items():
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or not p.parts:
            raise ValueError("Invalid bundle path")
        if digest(directory / p) != expected:
            raise ValueError("Checksum mismatch: " + name)
    return manifest


def publish(root, source, files, metadata=None, checkpoint=lambda phase: None):
    """Copy and verify before publishing; retain every previously published bundle."""
    root, source = Path(root), Path(source)
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
    token = uuid.uuid4().hex
    pointer_tmp = root / (".pointer-" + token)
    try:
        hashes = {}
        for name in sorted(set(files)):
            p = Path(name)
            if p.is_absolute() or ".." in p.parts or name == "manifest.json":
                raise ValueError("Invalid source path")
            dest = stage / p
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / p, dest)
            hashes[p.as_posix()] = digest(dest)
        manifest = {"schema": 1, "files": hashes, "metadata": metadata or {}}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        checkpoint("copied")
        verify(stage)
        checkpoint("verified")
        published = root / ("snapshot-" + token)
        os.replace(stage, published)
        checkpoint("bundle_renamed")
        pointer = {"snapshot": published.name, "manifest_sha256": digest(published / "manifest.json")}
        with pointer_tmp.open("w") as f:
            json.dump(pointer, f)
            f.flush()
            os.fsync(f.fileno())
        checkpoint("pointer_written")
        os.replace(pointer_tmp, root / "CURRENT.json")
        return published
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if pointer_tmp.exists():
            pointer_tmp.unlink()


def current(root):
    root = Path(root)
    pointer = json.loads((root / "CURRENT.json").read_text())
    name = pointer["snapshot"]
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Invalid snapshot pointer")
    directory = root / name
    if digest(directory / "manifest.json") != pointer["manifest_sha256"]:
        raise ValueError("Manifest checksum mismatch")
    verify(directory)
    return directory


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args = p.parse_args()
    print(current(args.root))
