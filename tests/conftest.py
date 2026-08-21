import pytest
import tiktoken

#: Every encoding the code under test asks ``tiktoken`` for.
TIKTOKEN_ENCODINGS = ("o200k_base",)


@pytest.fixture(scope="session", autouse=True)
def warm_tiktoken_cache() -> None:
    """Load the tiktoken BPE files before any network guard is installed.

    ``tiktoken.get_encoding`` downloads its BPE file with ``requests.get`` on a
    cold cache and memoises the result in ``tiktoken.registry`` for the rest of
    the process. Locally the cache is already warm, so the download never
    happens; on a fresh CI runner it does — and lands on the per-test
    ``requests.get`` patch, failing every test that estimates tokens with a
    misleading "live HTTP call to OpenRouter" error.

    This fixture is session-scoped, so pytest sets it up before the
    function-scoped guards, and the download (if any) happens exactly once
    while real networking is still available.
    """
    for encoding in TIKTOKEN_ENCODINGS:
        tiktoken.get_encoding(encoding)
