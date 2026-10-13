"""Helpers for the within-region train/test splits (notebook `2.3 Within_region`).

The splits themselves are computed by `prepare_within_region_pairs_from_cache` and
`prepare_within_region_random_splits` (`xreg_dataset.py`); these helpers manage
their checkpoints and merge them into one set of keys.
"""

import pickle
from pathlib import Path


def purge_codes_from_checkpoint(checkpoint_path, codes_to_purge):
    """Remove specific region codes from a split checkpoint so they get recomputed,
    while leaving everything else cached."""
    if not codes_to_purge or not Path(checkpoint_path).exists():
        return
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)
    res_by_code = ckpt.get("res_by_code", {})
    split_info_by_code = ckpt.get("split_info_by_code", {})
    failed_codes = ckpt.get("failed_codes", [])

    removed = []
    for code in codes_to_purge:
        if code in res_by_code:
            del res_by_code[code]
            removed.append(code)
        if code in split_info_by_code:
            del split_info_by_code[code]
        if code in failed_codes:
            failed_codes.remove(code)

    with open(checkpoint_path, "wb") as f:
        pickle.dump(
            {
                "res_by_code": res_by_code,
                "split_info_by_code": split_info_by_code,
                "failed_codes": failed_codes,
            },
            f,
        )
    if removed:
        print(f"  Purged {removed} from {checkpoint_path} for recompute")


def _rename_legacy_pad_keys(val):
    """Rename 'head_pad'/'tail_pad' (old caches) to 'months_head_pad'/'months_tail_pad'."""
    if "head_pad" in val and "months_head_pad" not in val:
        val["months_head_pad"] = val.pop("head_pad")
    if "tail_pad" in val and "months_tail_pad" not in val:
        val["months_tail_pad"] = val.pop("tail_pad")


def combine_within_region_splits(
    res_gl, split_info_gl, res_id, split_info_id, res_random, split_info_random
):
    """
    Merge the glacier-level (GL), measurement-level (ID) and random splits into
    one dict of datasets and one dict of split info, keyed by region code.

    Codes that have both a GL and an ID split (e.g. CA_3, CA_4, ALA) get the
    suffixes '_GL' and '_ID'; codes with only one split keep their plain code
    (SJM only has an ID split). Random splits are added as '<code>_rand<i>'.

    Also renames the old 'head_pad'/'tail_pad' keys of cached splits in place,
    so the (slow) split computation does not need to be rerun.

    Returns
    -------
    (res_within_by_code, split_info_by_code)
    """
    overlap_codes = set(res_gl) & set(res_id)

    res_within_by_code, split_info_by_code = {}, {}
    for code, val in res_gl.items():
        res_within_by_code[f"{code}_GL" if code in overlap_codes else code] = val
    for code, val in res_id.items():
        res_within_by_code[f"{code}_ID" if code in overlap_codes else code] = val
    for code, val in split_info_gl.items():
        split_info_by_code[f"{code}_GL" if code in overlap_codes else code] = val
    for code, val in split_info_id.items():
        split_info_by_code[f"{code}_ID" if code in overlap_codes else code] = val

    for code, res_list in res_random.items():
        for split_idx, val in enumerate(res_list):
            res_within_by_code[f"{code}_rand{split_idx}"] = val
    for code, split_info_list in split_info_random.items():
        for split_idx, val in enumerate(split_info_list):
            split_info_by_code[f"{code}_rand{split_idx}"] = val

    for d in (res_within_by_code, res_gl, res_id):
        for val in d.values():
            _rename_legacy_pad_keys(val)
    for res_list in res_random.values():
        for val in res_list:
            _rename_legacy_pad_keys(val)

    return res_within_by_code, split_info_by_code
