"""tinyllm -- an end-to-end LLM lifecycle, from random init to VS Code.

Import policy
-------------
This package spans two tiers with different dependencies:

  * ``config`` is stdlib-only and imports anywhere.
  * ``tokenizer``, ``data``, ``model_scratch``, ``parity``, ``train``, ``sft``,
    ``evaluate`` and ``export`` require torch/transformers and are Colab-only.

Nothing is eagerly imported here, so ``from tinyllm.config import model_cfg``
works on the local Windows tier, which has no PyTorch by design.
"""

from tinyllm import config

__all__ = ["config"]
__version__ = "0.1.0"
