---
created: 2026-06-04
author: agent
type: agent
status: final
tags: [daily-log, cache-architecture, pre-encode, shard-aware-shuffle, lru-cache, pipeline-redesign]
---

# Cache 架構設計：Pre-Encode 兩階段管線

## What

完成 `cache.py` 的完整架構設計，包含 shard 存儲格式、LRU shard cache、以及 **shard-aware shuffle sampler**。解決 random shuffle 會導致 LRU hit rate ~2.8% 的崩潰問題。

## Why

當前訓練管線每 batch 都要跑 DINOv3 + VAE.encode（2870 萬次重複運算），VRAM 和 I/O 是根本瓶頸。

隨機 shuffle + LRU 的設計會崩潰：
- 288 shards × 1000 images，LRU 只留 8 片 → hit rate ≈ 8/288 ≈ 2.8%
- 幾乎每次 __getitem__ 都要讀 HDD → 和不用 cache 沒兩樣

需要 **shard-aware shuffle**：先 shuffle shard 順序，再 shuffle shard 內順序 → hit rate ≈ 97.2%，每 125 batches 只 miss 一次。

## How

### Shard 存儲格式

```
cache/
├── manifest.json            # 全局元數據（版本、哈希、模型版本）
├── shard_00000/
│   ├── patch_tokens.pt      # [N, 1024, 384] float16 ≈ 7.6 MB
│   ├── gt_latents.pt        # [N, 16, 64, 64] float16 ≈ 1.3 MB
│   └── meta.json            # 圖片路徑、shard 資訊
├── shard_00001/
│   ├── patch_tokens.pt
│   ├── gt_latents.pt
│   └── meta.json
└── ...
```

**決策理由：**

| 決策 | 選項 A | 選項 B | 選擇 | 理由 |
|------|--------|--------|------|------|
| 存儲格式 | 單一 530GB 文件 | 分 shard 目錄 | ✅ 分 shard | 可恢復、可並行、可增量更新 |
| 精度 | float32 | float16 | ✅ float16 | 節省 50% 空間+I/O，過 fit 實驗精度夠用 |
| 版本控制 | 無 | 完整 manifest | ✅ manifest | 檢測 dataset/模型變化 |

### manifest.json 格式

```json
{
  "version": 1,
  "dataset_root": "/data/danbooru-images/danbooru-images",
  "dataset_hash": "sha256:abc123...",
  "model_versions": {
    "dino": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vae": "advokat/AnimePro-FLUX"
  },
  "total_shards": 288,
  "shard_count": 288,
  "created_at": "2026-06-04T12:00:00Z"
}
```

**失效檢查：**
1. 讀取 manifest
2. 計算當前 dataset root 的 sha256（所有路徑排序後 hash）
3. 比較 hash → 不同則需重編
4. 比較 model_versions → 不同則需重編

### 核心資料結構

```python
@dataclass
class ShardCacheConfig:
    """Cache pipeline configuration."""
    shard_size: int = 1000          # 每個 shard 的圖片數
    cache_dir: Path                 # cache 根目錄
    max_cached_shards: int = 8      # 記憶體保留的 shard 數
    dtype: torch.dtype = torch.float16  # 存儲精度

@dataclass
class ShardMeta:
    """Per-shard metadata stored in meta.json."""
    shard_id: int
    n_images: int
    image_paths: list[str]          # 原始路徑，供 debug
```

### ShardAwareSampler — 兩層 shuffle

```python
class ShardAwareSampler(Sampler[int]):
    """Two-level shuffle sampler.
    
    Level 1: shuffle shard order
    Level 2: shuffle within each shard (persistent per epoch)
    
    Usage:
        sampler = ShardAwareSampler(dataset, shard_size=1000, seed=42)
        loader = DataLoader(dataset, batch_size=8, sampler=sampler)
        # ⚠️ shuffle=False — 我們自己已經打亂了
    """
    def __init__(
        self,
        dataset: CachedDataset,
        shard_size: int = 1000,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.shard_size = shard_size
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        
        # Level 1: 打亂 shard 順序
        shard_ids = list(range(self.dataset.shard_count))
        rng.shuffle(shard_ids)
        
        # Level 2: 依序产出每個 shard 內的樣本
        for shard_id in shard_ids:
            shard_rng = random.Random(self.seed + shard_id)
            offsets = list(range(self.shard_size))
            shard_rng.shuffle(offsets)
            for offset in offsets:
                yield self.dataset._idx(shard_id, offset)

    def __len__(self) -> int:
        return self.dataset.__len__()
```

**關鍵設計：**
- `shuffle=True` 要設為 `False`（DataLoader 參數），因為我們已經自己 shuffle 了
- 每個 epoch 重新 shuffle shard 順序，但每個 shard 內順序用 `seed + shard_id` 固定
- 跨 epoch 的 shard 順序不同，但 shard 內順序一致（方便 debug）

### ShardCache — LRU + 容量控制

```python
class ShardCache:
    """LRU cache backed by OrderedDict. Per-Dataset instance (per-worker)."""

    def __init__(self, max_shards: int = 8):
        self._cache: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = (
            OrderedDict()
        )
        self.max_shards = max_shards
        self.hits = 0
        self.misses = 0

    def get(self, shard_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)  # LRU: 移到最後=最近用
            self.hits += 1
            return self._cache[shard_id]
        self.misses += 1
        return None

    def put(self, shard_id: int, tokens: torch.Tensor, latents: torch.Tensor):
        self._cache[shard_id] = (tokens, latents)
        self._cache.move_to_end(shard_id)
        while len(self._cache) > self.max_shards:
            # evict LRU (最前面)
            _, (evict_tokens, evict_latents) = self._cache.popitem(last=False)
            del evict_tokens, evict_latents  # 釋放 memory

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._cache),
            "max": self.max_shards,
            "hit_rate": f"{self.hits/max(total,1)*100:.1f}%",
        }
```

### CachedDataset — O(1) index 查找

```python
class CachedDataset(Dataset):
    """PyTorch Dataset backed by pre-encoded shards.
    
    Index mapping: {global_idx → (shard_id, local_offset)}
    Lookup: O(1) dict lookup → LRU cache → return tensor slice
    
    Shuffle 由 ShardAwareSampler 處理，__getitem__ 不做任何 shuffle。
    """
    def __init__(self, config: ShardCacheConfig):
        self.config = config
        self.shard_size = config.shard_size
        self._build_index_map()
        self._cache = ShardCache(config.max_cached_shards)

    def _build_index_map(self):
        """建立 {global_idx → (shard_id, offset)} 倒排表。
        
        不需要打亂——shuffle 由 Sampler 處理。
        """
        manifest = self._load_manifest()
        self.shard_count = manifest["total_shards"]
        
        self._index_map: dict[int, tuple[int, int]] = {}
        
        shard_id = 0
        for path in sorted((config.cache_dir).glob("shard_*")):
            meta = json.loads((path / "meta.json").read_text())
            n = meta["n_images"]
            for offset in range(n):
                self._index_map[shard_id * self.shard_size + offset] = (shard_id, offset)
            shard_id += 1

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard_id, offset = self._index_map[idx]

        # 檢查 cache
        cached = self._cache.get(shard_id)
        if cached is not None:
            tokens, latents = cached
            tokens = tokens[offset:offset+1].to(torch.float32)
            latents = latents[offset:offset+1].to(torch.float32)
            return tokens, latents

        # Cache miss — 從 HDD 載入整個 shard
        shard_path = self.config.cache_dir / f"shard_{shard_id:05d}"
        tokens = torch.load(shard_path / "patch_tokens.pt").to(torch.float32)
        latents = torch.load(shard_path / "gt_latents.pt").to(torch.float32)

        # 進 cache（整個 shard 留下供後續 batch 用）
        self._cache.put(shard_id, tokens, latents)

        tokens = tokens[offset:offset+1]
        latents = latents[offset:offset+1]
        return tokens, latents
```

**關鍵設計：**
- index map 不打亂，shuffle 由 `ShardAwareSampler` 處理
- `__getitem__` 是純 O(1) 查找 + cache lookup
- 載入 shard 後轉 float32（训练需要精度），存儲用 float16
- 整個 shard 留下 cache 供後續連續 batch 用

### 完整使用方式

```python
from torch.utils.data import DataLoader

config = ShardCacheConfig(
    shard_size=1000,
    cache_dir=Path("~/hdd/cache"),
    max_cached_shards=8,
    dtype=torch.float16,
)

dataset = CachedDataset(config)
sampler = ShardAwareSampler(dataset, shard_size=1000, seed=42)

loader = DataLoader(
    dataset,
    batch_size=8,
    sampler=sampler,       # ← 不要 shuffle=True!
    num_workers=4,
    collate_fn=lambda batch: (
        torch.stack([b[0].squeeze(0) for b in batch]),   # tokens
        torch.stack([b[1].squeeze(0) for b in batch]),   # latents
    ),
)
```

### 效能預估

```
shard_size=1000, batch_size=8 → 每 shard = 125 batches
LRU(8 shards):

┌──────────┬───────────┬──────────────┬──────────────────┐
│ Epoch    │ Shard seq │ Miss freq    │ Est. shard I/O   │
├──────────┼───────────┼──────────────┼──────────────────┤
│ #1       │ 288 unique│ every 125 b. │ 8 loads/epoch   │
│ #2~#100  │ same      │ every 125    │ 8 loads/epoch   │
└──────────┴───────────┴──────────────┴──────────────────┘

總 I/O: 8 shards × 1.84 GB × 100 epochs = 1.47 TB
vs 傳統: 287038 × 100 = 28.7M 次完整編碼 (DINOv3 + VAE.encode)
```

## Follow-up

- [ ] 實作 `cache.py`（ShardAwareSampler + ShardCache + CachedDataset）
- [ ] 實作 `encode.py`（pre-encode pipeline 入口）
- [ ] 實作 pre-encode 的並行和斷點續傳
- [ ] 重構 `trainer.py`（去掉 DINO/VAE 依賴，專為 cached training 優化）
- [ ] 重構 `train.py`（改用 cache 模式）
- [ ] 執行 pre-encode（預估 ~2 小時）
- [ ] 比較 pre-encode vs 傳統 training 的速度差異

## References

- [13 Training Pipeline Restructure](./13_2026_06_03_agent_training-pipeline-restructure.md) — 當前架構
- [15 Pre-Encode Cache Pipeline](./15_2026_06_04_human_pre-encode-cache-pipeline.md) — 大重構決定
