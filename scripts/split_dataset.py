from pathlib import Path
import random
from collections import defaultdict

import numpy as np

NPZ_DIR = Path("data/AMASS/npz")
OUTPUT_DIR = Path("data/AMASS")

split_ratio = {"train": 0.7, "val": 0.15, "test": 0.15}


def subject_key(path: Path) -> str:
    try:
        with np.load(path, allow_pickle=True) as d:
            if "subject_id" in d.files and "subset" in d.files:
                return f"{d['subset'].item()}_{d['subject_id'].item()}"
    except Exception as e:
        print(f"Could not read metadata from {path.name}: {e}")
    return extract_subject_id(path)


def extract_subject_id(path: Path) -> str:
    name = path.stem
    parts = name.split("_")
    
    if len(parts) < 2:
        return name
    
    dataset = parts[0]
    
    if dataset == "CMU" and len(parts) >= 2 and parts[1].isdigit():
        return f"CMU_{parts[1]}"
    
    elif dataset == "Transitions" and len(parts) >= 3 and parts[1] == "mocap":
        return f"Transitions_{parts[2]}"
    
    elif dataset == "KIT":
        action_parts = parts[1:]
        if not action_parts:
            return name
        
        last_part = action_parts[-1]
        if last_part.isdigit():
            action_parts = action_parts[:-1]
        else:
            cleaned = ''.join(c for c in last_part if not c.isdigit())
            if cleaned and cleaned != last_part:
                action_parts[-1] = cleaned
        
        if not action_parts:
            return name
        
        action = "_".join(action_parts)
        return f"KIT_{action}"
    
    return name

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    subject_to_files = defaultdict(list)
    for npz_path in sorted(NPZ_DIR.glob("*.npz")):
        try:
            subject_to_files[subject_key(npz_path)].append(npz_path)
        except Exception as e:
            print(f"Failed on {npz_path.name}: {e}")

    by_subset = defaultdict(set)
    for subj in subject_to_files:
        by_subset[subj.split("_")[0]].add(subj)
    print("Subjects found per subset:")
    for s, v in sorted(by_subset.items()):
        print(f"  {s}: {len(v)}")

    subjects = sorted(subject_to_files.keys())
    random.shuffle(subjects)

    n_total = len(subjects)
    n_train = int(split_ratio["train"] * n_total)
    n_val = int(split_ratio["val"] * n_total)

    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:]

    splits = {"train": [], "val": [], "test": []}
    for split_name, subject_list in zip(["train", "val", "test"], [train_subjects, val_subjects, test_subjects]):
        for subj in subject_list:
            splits[split_name].extend(subject_to_files[subj])

    for split_name, paths in splits.items():
        out_file = OUTPUT_DIR / f"{split_name}.txt"
        with out_file.open("w") as f:
            for p in sorted(paths):
                f.write(str(p) + "\n")
        print(f"{len(paths)} written to {out_file}")

    print("\nSplit summary by subject count:")
    subj_of = {p: s for s, ps in subject_to_files.items() for p in ps}
    for split_name, paths in splits.items():
        unique_subjects = {subj_of[p] for p in paths}
        print(f"  {split_name.upper()}: {len(unique_subjects)} subjects, {len(paths)} files")

    all_subjects = [set(), set(), set()]
    for i, name in enumerate(["train", "val", "test"]):
        all_subjects[i] = {subj_of[p] for p in splits[name]}
    overlap = (all_subjects[0] & all_subjects[1]) | (all_subjects[0] & all_subjects[2]) | (all_subjects[1] & all_subjects[2])
    print(f"\nSubject overlap between splits: {len(overlap)}  (must be 0)")


if __name__ == "__main__":
    main()
