import asyncio

from memory.conversation_memory import MemoryManager


def test_long_term_memory_can_be_disabled_for_local_demo():
    manager = MemoryManager(
        redis_url="redis://127.0.0.1:6379/0",
        api_key="test-key",
        enable_long_term_memory=False,
    )

    assert asyncio.run(manager._search_episodic("user", "conv", "query")) == []
    assert asyncio.run(manager._get_profile("user")) == {}
    asyncio.run(manager.close())
