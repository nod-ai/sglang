from __future__ import annotations

"""Cache for chunked prefill, used when RadixCache is disabled."""

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class ChunkCacheEntry:
    def __init__(self, rid: str, value: torch.Tensor):
        self.rid = rid
        self.value = value


class ChunkCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        page_size: int,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = page_size

    def reset(self):
        pass

    def match_prefix(self, rid: int, key: List[int]) -> Tuple[List[int], int]:
        if rid not in self.entries:
            return [], None

        entry = self.entries[rid]
        max_prefix_len = len(key)
        return entry.value[:max_prefix_len], entry

    def cache_finished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        if token_ids is None:
            token_id_len = len(req.origin_input_ids) + len(req.output_ids) - 1
        else:
            token_id_len = len(token_ids)

    def cache_finished_req(self, req: Req):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx,
            # For decode server: if req.output_ids is empty, we want to free all req.origin_input_ids
            : len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0),
        ]
        self.req_to_token_pool.free(req.req_pool_idx)
        self.token_to_kv_pool_allocator.free(kv_indices)

    def cache_unfinished_req(self, req: Req):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ]

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        req.prefix_indices = kv_indices

    def insert(self):
        raise NotImplementedError()

    def evict(self, num_tokens: int):
        pass

    def inc_lock_ref(self, node: Any):
        return 0

    def dec_lock_ref(self, node: Any):
        return 0

    def evictable_size(self):
        return 0

    def protected_size(self):
        return 0
