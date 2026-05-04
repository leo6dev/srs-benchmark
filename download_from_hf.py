# from huggingface_hub import HfApi
# import fnmatch
#
# api = HfApi(token="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")
# repo_files = api.list_repo_files("open-spaced-repetition/anki-revlogs-10k", repo_type="dataset")
# print(f"Total repo entries: {len(repo_files)}")
#
# # Build the same patterns for preview (example with first 50 users)
# top_dirs = ["revlogs", "cards", "decks"]
# preview_patterns = [f"{d}/user_id={i}/*.parquet" for d in top_dirs for i in range(1, 51)]
#
# matched = []
# for p in preview_patterns:
#     # fnmatch.translate isn't necessary because glob-style '**' isn't directly supported by fnmatch,
#     # but the library uses unix shell-style. To handle '**' simply replace it with '*' for preview.
#     norm_pat = p.replace("/**/", "/*/")  # rough normalization for preview purposes
#     matches = [f for f in repo_files if fnmatch.fnmatch(f, norm_pat)]
#     if matches:
#         matched.extend(matches)
#
# print("Sample matched paths (up to 30):")
# for path in matched[:30]:
#     print(" ", path)

from huggingface_hub import snapshot_download  # type: ignore
from pathlib import Path

download_dir = Path("../anki-revlogs-10k")
download_dir.mkdir(parents=True, exist_ok=True)

print(download_dir)

# choose how many users you want
N = 500  # e.g. 500..1000 to stay within ~2-3GB

top_dirs = ["revlogs", "cards", "decks"]  # top-level folders in the repo
patterns = []
for d in top_dirs:
    for i in range(2000, 2000+N + 1):
        # match any parquet files under the user's folder (any nesting)
        patterns.append(f"{d}/user_id={i}/*.parquet")

# remove duplicates (not necessary but tidy)
patterns = list(dict.fromkeys(patterns))

print(f"Generated {len(patterns)} patterns (for {N} users across {top_dirs}).")

snapshot_download(
    repo_id="open-spaced-repetition/anki-revlogs-10k",
    repo_type="dataset",
    allow_patterns=patterns,
    local_dir=str(download_dir),
    token="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"
)


# #
# # from datasets import load_dataset
# #
# # dataset = load_dataset(
# #     "open-spaced-repetition/anki-revlogs-10k",
# #     streaming=True,
# #     token="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"
# # )
# #
# # small_sample = []
# # for i, example in enumerate(dataset["train"]):
# #     small_sample.append(example)
# #     if i >= 100:   # stop after some amount
# #         print(i)
# #         break