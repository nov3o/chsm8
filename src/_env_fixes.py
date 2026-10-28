"""
Environment workarounds — import this before `transformers` anywhere.

This node's system-installed torchvision/torchaudio (under
/usr/local/lib/python3.12/dist-packages, leaked into our venv via
--system-site-packages) are ABI-mismatched against the installed torch
build (torch is a newer/differently-built version than these were compiled
for), so importing either raises at C-extension-registration time
(`RuntimeError: operator torchvision::nms does not exist`, then similarly
for torchaudio's compiled lib). `transformers` unconditionally imports both
inside its core `modeling_utils`/`processing_utils` chain (to register
vision/audio loss classes we never use), so ANY `.from_pretrained(...)` call
crashes without this fix — even for plain text models like Qwen or T5.

We don't use vision or audio anywhere in this project, so this installs a
meta_path finder that intercepts any `torchvision`/`torchaudio` import (at
any submodule depth) and returns an inert stub instead of touching the
real, broken packages on disk.
"""
import sys
import types
import importlib.abc
import importlib.machinery


class _DummyAttr:
    def __call__(self, *a, **k):
        return None

    def __getattr__(self, name):
        return _DummyAttr()


class _BlockedLoader(importlib.abc.Loader):
    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__path__ = []  # makes it look like a package so submodule imports keep working
        m.__getattr__ = lambda name: _DummyAttr()
        return m

    def exec_module(self, module):
        pass


class _BlockedFinder(importlib.abc.MetaPathFinder):
    BLOCKED = ("torchvision", "torchaudio")

    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in self.BLOCKED:
            return importlib.machinery.ModuleSpec(name, _BlockedLoader(), is_package=True)
        return None


import os
# LLM_CHESS_NO_ENV_FIXES=1: environments with working torchvision (e.g. the vLLM container, whose spawned workers
# re-import our modules) skip the stubs.
if os.environ.get("LLM_CHESS_NO_ENV_FIXES") != "1" and not any(isinstance(f, _BlockedFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _BlockedFinder())
