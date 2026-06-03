---
created: 2026-06-03
author: FlyDogDaDa
type: agent
status: final
tags: [daily-log, dataset, tagged-anime-illustrations, nas]
---

# Tagged Anime Illustrations Dataset Discovery

## What

- Located the "tagged-anime-illustrations" dataset on NAS
- Documented the dataset structure for future reference

## Why

User needed to find the dataset location and understand its structure for DeepShader project development.

## Background

Source: [Kaggle - tagged-anime-illustrations](https://www.kaggle.com/datasets/mylesoneill/tagged-anime-illustrations)

This dataset combines two anime image sources:

### 1. Danbooru2017
- **Source**: [Danbooru](https://www.danbooru.donmai.us/) — the best known anime image booru
- **Scale**: ~1.9TB, 2.94M images, 77.5M tag instances, 333K defined tags (~26.3 tags/image)
- **Time span**: May 24, 2005 – Dec 31, 2017
- **In our copy**: 337,038 images in normalized 512×512px JPG, with full metadata in JSON
- **SFW subset**: 2.23M images (original dataset offers 241GB SFW downscaled version)

**Research paper**: *Danbooru2017: A large-scale crowdsourced and tagged anime illustration dataset*
(GWern: https://www.gwern.net/Danbooru2017)

### 2. MoeImouto Faces
- **Source**: [Anime Face Character Dataset](http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/) (defunct)
- **Content**: Cropped illustrated character faces (PNG) + CSV position files
- **History**: Previously used in GAN research
- **Face detection tool**: https://github.com/nagadomi/lbpcascade_animeface

### Purpose
Rich large-scale classification/tagging & learned embeddings, transferability testing of CV techniques to anime-style images, archival backup, conditional image generation, style transfer.

---

## How

1. Searched through `~/hdd` (symlink to `/mnt/hdd/b11223209`) — not found
2. Identified NAS mount at `/mnt/nas` (NFS from `192.168.0.19`)
3. Found dataset in NAS backup path:
   ```
   /mnt/nas/PC_Data/NAS PC_data/2025_02_10_old_hdd/B11223209_Vincent/Tagged_Anime_Illustration/
   ```
4. Documented the 3-part structure:
   - **danbooru-images/**: 337,038 JPG images split into numbered subfolders (0000~0029+)
   - **danbooru-metadata/**: Danbooru tags in JSON format (monthly partitioned files)
   - **moeimouto-faces/**: Character face crops (PNG) + CSV position files

## Follow-up

- Explore using this dataset for model training
- Check the `Tagged_Anime_Illustration_preTansform` folder for any preprocessed version

## References

- Kaggle dataset: [tagged-anime-illustrations](https://www.kaggle.com/datasets/mylesoneill/tagged-anime-illustrations)
- Dataset source: [Danbooru](https://www.danbooru.donmai.us/) + [MoeImouto](https://moeimouto.com/)
