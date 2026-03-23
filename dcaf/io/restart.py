from dcaf.io.output import (
    load_latest_snapshot,
    load_snapshot_by_index,
    find_latest_output_folder,
)


def formation_finished_at_time(model_time, framework):
    last_time = framework.get_last_formation_time()

    if last_time is None:
        return True

    return model_time >= last_time


def validate_resume_after_formation(model_time, framework):
    if not formation_finished_at_time(model_time, framework):
        last_time = framework.get_last_formation_time()
        raise ValueError(
            "Resume only supported after star formation is finished. "
            f"Snapshot time is {model_time.in_(last_time.unit)}, "
            f"but last formation time is {last_time.in_(last_time.unit)}."
        )


def get_resume_state(snapshot_index=None,
                     framework=None,
                     output_folder="dcaf_output",
                     snapshot_basename="stars_"):

    resume_folder = find_latest_output_folder(output_folder)

    if snapshot_index is None:
        state = load_latest_snapshot(
            output_folder=resume_folder,
            snapshot_basename=snapshot_basename,
        )
    else:
        state = load_snapshot_by_index(
            snapshot_index,
            output_folder=resume_folder,
            snapshot_basename=snapshot_basename,
        )

    state["output_folder"] = resume_folder

    if framework is not None:
        validate_resume_after_formation(state["model_time"], framework)

    return state
