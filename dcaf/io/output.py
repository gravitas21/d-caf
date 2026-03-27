import glob
import os
import re

from amuse.io import read_set_from_file


def snapshot_filename(snapshot_index, source_folder="dcaf_output", snapshot_basename="stars_"):
    filename = f"{snapshot_basename}{snapshot_index:03d}.amuse"
    return os.path.join(source_folder, filename)


def snapshot_index_from_path(path, snapshot_basename="stars_"):
    name = os.path.basename(path)
    m = re.match(rf"{re.escape(snapshot_basename)}(\d+)\.amuse$", name)
    if m is None:
        raise ValueError(
            f"File '{path}' does not match snapshot pattern "
            f"'{snapshot_basename}###.amuse'"
        )
    return int(m.group(1))


def find_snapshot_files(source_folder="dcaf_output", snapshot_basename="stars_"):
    pattern = os.path.join(source_folder, f"{snapshot_basename}*.amuse")
    files = glob.glob(pattern)

    snapshots = []
    for path in files:
        try:
            idx = snapshot_index_from_path(path, snapshot_basename=snapshot_basename)
            snapshots.append((idx, path))
        except ValueError:
            pass

    snapshots.sort(key=lambda x: x[0])
    return snapshots


def find_latest_snapshot(source_folder="dcaf_output", snapshot_basename="stars_"):
    snapshots = find_snapshot_files(
        source_folder=source_folder,
        snapshot_basename=snapshot_basename,
    )

    if len(snapshots) == 0:
        raise FileNotFoundError(
            f"No snapshots found in '{source_folder}' with basename '{snapshot_basename}'"
        )

    return snapshots[-1]

def find_latest_output_folder(base_output_folder="dcaf_output"):
    parent = os.path.dirname(base_output_folder.rstrip("/"))
    if parent == "":
        parent = "."
    stem = os.path.basename(base_output_folder.rstrip("/"))

    candidates = glob.glob(os.path.join(parent, stem + "*"))

    best_seg = None
    best_path = None

    for path in candidates:
        name = os.path.basename(path.rstrip("/"))
        if name == stem:
            seg = 0
        else:
            m = re.fullmatch(rf"{re.escape(stem)}_(\d+)", name)
            if m is None:
                continue
            seg = int(m.group(1))

        if not os.path.isdir(path):
            continue

        if best_seg is None or seg > best_seg:
            best_seg = seg
            best_path = path

    if best_path is None:
        raise FileNotFoundError(
            f"No output folders found matching '{base_output_folder}'"
        )

    return best_path


def load_snapshot(path):
    stars = read_set_from_file(path, format="amuse")

    if not hasattr(stars.collection_attributes, "model_time"):
        raise ValueError(
            f"Snapshot '{path}' does not contain collection_attributes.model_time"
        )

    model_time = stars.collection_attributes.model_time

    return {
        "path": path,
        "snapshot_index": None,
        "stars": stars,
        "model_time": model_time,
    }


def load_snapshot_by_index(snapshot_index, source_folder="dcaf_output", snapshot_basename="stars_"):
    path = snapshot_filename(
        snapshot_index,
        source_folder=source_folder,
        snapshot_basename=snapshot_basename,
    )
    state = load_snapshot(path)
    state["snapshot_index"] = snapshot_index
    return state


def load_latest_snapshot(source_folder="dcaf_output", snapshot_basename="stars_"):
    snapshot_index, path = find_latest_snapshot(
        source_folder=source_folder,
        snapshot_basename=snapshot_basename,
    )

    state = load_snapshot(path)
    state["snapshot_index"] = snapshot_index
    return state
