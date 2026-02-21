Common Errors
=============

NLE build failure
-----------------

**Symptom:** ``ImportError: cannot import name 'nethack'`` or build errors
during ``pip install nle``.

**Cause:** Missing system build dependencies for the NetHack Learning
Environment.

**Fix:**

.. code-block:: bash

   sudo apt-get install -y build-essential cmake libncurses-dev bison flex

Then reinstall: ``pip install nle --no-cache-dir``

API rate limit exceeded
------------------------

**Symptom:** ``openai.RateLimitError`` or ``anthropic.RateLimitError``
during evaluation.

**Cause:** Too many parallel workers hitting the API concurrently.

**Fix:** Reduce ``parallel_workers`` in the config (default: 4).  For
high-throughput evaluation, use a local vLLM endpoint instead.

History window too large
-------------------------

**Symptom:** ``ContextLengthExceededError`` from the LLM API.

**Cause:** Long observation histories exceed the model's context window.

**Fix:** Reduce ``history_window`` (default: 4) or switch to a model
with a larger context (e.g. ``claude-3-5-sonnet`` → 200k tokens).
