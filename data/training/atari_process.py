"""Prepare training data of the JustAct encoder: Atari frames and manifest.

Source: the public Hugging Face datasets
    edbeeching/prj_gia_dataset_atari_2B_atari_<game>_1111   (file dataset.npy)
each holding a dict with
  - observations: (N, 1, 4, 84, 84) uint8, 4 stacked frames
  - actions:      (N, 1) float64, discrete action index
  - dones:        (N, 1) float64, episode-end flag

The most recent frame of each stack (index 3) gives the frame sequence, which
is split into episodes at `dones`. Episodes shorter than --min_episode_len are
dropped, longer ones are cut to --max_episode_len, and the first
--max_episodes_per_game episodes of each game are kept.

Output layout under <data_root>/<subdir>/:
    frames/<game>/episode_<N:05d>/frame_<T:04d>.png   (84x84 grayscale)
    actions/atari_<game>_ep<N:05d>/actions.npy         (T,) int32
    actions/index.json
    manifest_atari.json   (frame paths relative to data_root)

Usage:
    python atari_process.py --data_root <DATA_ROOT> --hf_cache_dir <HF_CACHE> \
        --games pong breakout spaceinvaders mspacman seaquest \
        --max_episodes_per_game 1000
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download


# Atari action meanings (full set, 18 actions)
ATARI_ACTIONS = [
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN",
    "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT",
    "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE",
    "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE",
]

# Game name -> short text description stored in the manifest
GAME_DESCRIPTIONS = {
    "pong": "play Pong",
    "breakout": "play Breakout, break bricks with the paddle and ball",
    "space_invaders": "play Space Invaders, shoot down alien invaders",
    "ms_pacman": "play Ms. Pacman, eat dots and avoid ghosts",
    "seaquest": "play Seaquest, shoot enemies and rescue divers",
    "asterix": "play Asterix, collect items and avoid enemies",
    "qbert": "play Q*bert, hop on cubes to change colors",
    "alien": "play Alien, navigate corridors and shoot aliens",
}


def download_game(game: str, hf_cache_dir: str) -> str:
    """Download one game's dataset.npy from Hugging Face. Returns the local path."""
    repo = f"edbeeching/prj_gia_dataset_atari_2B_atari_{game}_1111"
    print(f"  Downloading {repo}...")
    return hf_hub_download(
        repo, "dataset.npy", repo_type="dataset", cache_dir=hf_cache_dir
    )


def process_game(game: str, dataset_path: str, output_dir: str, data_root: str,
                 min_episode_len: int = 20, max_episodes: int = 1000,
                 max_episode_len: int = 1000) -> tuple:
    """Extract the episodes of one game.

    Returns (tasks, action_index) with paths relative to data_root.
    """
    def rel(p):
        return os.path.relpath(p, data_root)

    print(f"  Loading {game} dataset.npy ...")
    arr = np.load(dataset_path, allow_pickle=True).item()

    obs = arr["observations"]   # (N, 1, 4, 84, 84) uint8
    actions = arr["actions"][:, 0].astype(np.int32)  # (N,)
    dones = arr["dones"][:, 0].astype(bool)  # (N,)

    n_total = len(obs)
    print(f"  {game}: {n_total} transitions")

    # Most recent frame of each 4-stack
    frames = obs[:, 0, 3]  # (N, 84, 84)

    # Split into episodes at `dones`
    episode_boundaries = [0]
    for i, d in enumerate(dones):
        if d:
            episode_boundaries.append(i + 1)
    if episode_boundaries[-1] != n_total:
        episode_boundaries.append(n_total)

    episodes = []
    for i in range(len(episode_boundaries) - 1):
        start = episode_boundaries[i]
        end = episode_boundaries[i + 1]
        ep_len = end - start
        if ep_len < min_episode_len:
            continue
        # Cap episode length
        if ep_len > max_episode_len:
            end = start + max_episode_len
            ep_len = max_episode_len
        episodes.append((start, end))

    print(f"  {game}: {len(episodes)} episodes (after filtering)")

    if max_episodes > 0:
        episodes = episodes[:max_episodes]

    frames_dir = os.path.join(output_dir, "frames", game)
    actions_dir = os.path.join(output_dir, "actions")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(actions_dir, exist_ok=True)

    tasks = []
    action_index = {}

    for ep_idx, (start, end) in enumerate(tqdm(episodes, desc=f"  {game} episodes")):
        ep_len = end - start
        task_id = f"atari_{game}_ep{ep_idx:05d}"
        ep_frames_dir = os.path.join(frames_dir, f"episode_{ep_idx:05d}")
        ep_action_dir = os.path.join(actions_dir, task_id)
        action_path = os.path.join(ep_action_dir, "actions.npy")

        # Already processed: reuse the frames on disk
        if os.path.exists(action_path) and os.path.isdir(ep_frames_dir):
            existing_pngs = [f for f in os.listdir(ep_frames_dir) if f.endswith(".png")]
            if len(existing_pngs) == ep_len:
                frame_paths = sorted(
                    os.path.join(ep_frames_dir, f) for f in existing_pngs
                )
                tasks.append(_build_task_entry(
                    task_id, game, [rel(p) for p in frame_paths], list(range(ep_len))
                ))
                action_index[task_id] = rel(action_path)
                continue

        os.makedirs(ep_frames_dir, exist_ok=True)
        os.makedirs(ep_action_dir, exist_ok=True)

        frame_paths = []
        for j in range(ep_len):
            frame_path = os.path.join(ep_frames_dir, f"frame_{j:04d}.png")
            if not os.path.exists(frame_path):
                Image.fromarray(frames[start + j], mode="L").save(frame_path)
            frame_paths.append(frame_path)

        ep_actions = actions[start:end]  # (ep_len,) int32
        np.save(action_path, ep_actions)

        tasks.append(_build_task_entry(
            task_id, game, [rel(p) for p in frame_paths], list(range(ep_len))
        ))
        action_index[task_id] = rel(action_path)

    return tasks, action_index


def _build_task_entry(task_id, game, frame_paths, frame_indices):
    return {
        "task_id": task_id,
        "scene_id": game,
        "task_type": "atari",
        "description": GAME_DESCRIPTIONS.get(game, f"play {game}"),
        "hdf5_file": "",
        "demo_key": "",
        "demo_idx": 0,
        "split": "atari",
        "target_frames": {
            "paths": frame_paths,
            "frame_indices": frame_indices,
            "n_frames": len(frame_paths),
        },
        "context_frames": {
            "paths": [frame_paths[0]] if frame_paths else [],
            "n_frames": 1,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Process Atari replay data into the training manifest format")
    parser.add_argument("--data_root", required=True,
                        help="Folder that holds all processed datasets")
    parser.add_argument("--subdir", default="atari_processed",
                        help="Atari output folder inside data_root")
    parser.add_argument("--hf_cache_dir", required=True,
                        help="Hugging Face cache folder for the downloaded dataset.npy files")
    parser.add_argument("--games", nargs="+", required=True,
                        help="Game names as used in the Hugging Face repo ids (e.g. pong breakout)")
    parser.add_argument("--max_episodes_per_game", type=int, default=1000)
    parser.add_argument("--min_episode_len", type=int, default=20)
    parser.add_argument("--max_episode_len", type=int, default=1000)
    args = parser.parse_args()

    output_dir = os.path.join(args.data_root, args.subdir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.hf_cache_dir, exist_ok=True)

    all_tasks = []
    all_action_index = {}

    for game in args.games:
        print(f"\n=== Game: {game} ===")
        try:
            dataset_path = download_game(game, args.hf_cache_dir)
        except Exception as e:
            print(f"  ERROR downloading {game}: {e}")
            continue

        tasks, action_index = process_game(
            game, dataset_path, output_dir, args.data_root,
            min_episode_len=args.min_episode_len,
            max_episodes=args.max_episodes_per_game,
            max_episode_len=args.max_episode_len,
        )
        all_tasks.extend(tasks)
        all_action_index.update(action_index)

    manifest = {
        "stage": 1,
        "dataset": "atari",
        "dataset_info": {
            "games": args.games,
            "n_tasks": len(all_tasks),
            "frame_size": [84, 84],
            "n_actions": len(ATARI_ACTIONS),
            "max_episodes_per_game": args.max_episodes_per_game,
        },
        "tasks": all_tasks,
    }

    manifest_path = os.path.join(output_dir, "manifest_atari.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Action index: task_id -> actions.npy path relative to data_root
    index_path = os.path.join(output_dir, "actions", "index.json")
    with open(index_path, "w") as f:
        json.dump(all_action_index, f, indent=2)

    from collections import Counter
    game_dist = Counter(t["scene_id"] for t in all_tasks)
    print(f"\nTotal tasks: {len(all_tasks)} -> {manifest_path}")
    for g, n in game_dist.most_common():
        print(f"  {g}: {n} episodes")


if __name__ == "__main__":
    main()
