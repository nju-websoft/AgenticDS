from __future__ import annotations

import argparse
import hashlib
import os
import threading
from functools import lru_cache
from pathlib import Path

import requests

from global_values import CACHE_DIR


LUCENE_VERSION = "9.12.2"
LUCENE_ARTIFACTS = ("lucene-core", "lucene-analysis-common")
LUCENE_JAR_DIR = Path(os.getenv("AGENTICDS_LUCENE_JAR_DIR", str(Path(CACHE_DIR) / "lucene" / LUCENE_VERSION)))
_JVM_LOCK = threading.Lock()


def install_lucene() -> None:
    """Download pinned Apache artifacts and verify Maven Central SHA-512 digests."""
    LUCENE_JAR_DIR.mkdir(parents=True, exist_ok=True)
    for artifact in LUCENE_ARTIFACTS:
        filename = f"{artifact}-{LUCENE_VERSION}.jar"
        url = f"https://repo.maven.apache.org/maven2/org/apache/lucene/{artifact}/{LUCENE_VERSION}/{filename}"
        checksum = requests.get(url + ".sha512", timeout=(10, 60))
        checksum.raise_for_status()
        expected = checksum.text.strip().split()[0].lower()
        path = LUCENE_JAR_DIR / filename
        if path.exists() and hashlib.sha512(path.read_bytes()).hexdigest() == expected:
            print(f"[lucene] verified {path}", flush=True)
            continue
        temporary = path.with_suffix(".jar.part")
        digest = hashlib.sha512()
        with requests.get(url, stream=True, timeout=(10, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != expected:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"SHA-512 mismatch for {filename}")
        temporary.replace(path)
        print(f"[lucene] installed {path}", flush=True)


@lru_cache(maxsize=1)
def lucene_classes() -> dict:
    import jpype

    with _JVM_LOCK:
        jars = [LUCENE_JAR_DIR / f"{name}-{LUCENE_VERSION}.jar" for name in LUCENE_ARTIFACTS]
        if any(not path.is_file() for path in jars):
            raise FileNotFoundError("Lucene jars are missing. Run: python -m utils.lucene_runtime --install")
        if not jpype.isJVMStarted():
            jpype.startJVM(
                f"-Xmx{os.getenv('AGENTICDS_LUCENE_JVM_HEAP', '4g')}",
                # Python's large immutable entity heap must not trigger full JVM/Python GC cycles.
                "-XX:+DisableExplicitGC",
                classpath=[str(path.resolve()) for path in jars],
                convertStrings=True,
            )
        version = str(jpype.JClass("org.apache.lucene.util.Version").LATEST)
        if version != LUCENE_VERSION:
            raise RuntimeError(f"Expected Lucene {LUCENE_VERSION}, found {version}; use a fresh process")
        names = {
            "Paths": "java.nio.file.Paths",
            "FSDirectory": "org.apache.lucene.store.FSDirectory",
            "WhitespaceAnalyzer": "org.apache.lucene.analysis.core.WhitespaceAnalyzer",
            "Document": "org.apache.lucene.document.Document",
            "StoredField": "org.apache.lucene.document.StoredField",
            "TextField": "org.apache.lucene.document.TextField",
            "Store": "org.apache.lucene.document.Field$Store",
            "IndexWriterConfig": "org.apache.lucene.index.IndexWriterConfig",
            "IndexWriter": "org.apache.lucene.index.IndexWriter",
            "DirectoryReader": "org.apache.lucene.index.DirectoryReader",
            "IndexSearcher": "org.apache.lucene.search.IndexSearcher",
            "BM25Similarity": "org.apache.lucene.search.similarities.BM25Similarity",
            "BooleanQuery": "org.apache.lucene.search.BooleanQuery",
            "BooleanQueryBuilder": "org.apache.lucene.search.BooleanQuery$Builder",
            "Occur": "org.apache.lucene.search.BooleanClause$Occur",
            "TermQuery": "org.apache.lucene.search.TermQuery",
            "Term": "org.apache.lucene.index.Term",
        }
        return {name: jpype.JClass(class_name) for name, class_name in names.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Install and verify the pinned Apache Lucene runtime.")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if args.install:
        install_lucene()
    lucene_classes()
    print(f"[lucene] ready version={LUCENE_VERSION} jars={LUCENE_JAR_DIR}", flush=True)


if __name__ == "__main__":
    main()
