import numpy as np
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import random as rd
import torch

import massbalancemachine as mbm


# -------------------------------- DATA LOADERS --------------
def make_loaders_TL(
    ds_tl,  # MBSequenceDatasetTL
    train_idx,
    val_idx,
    *,
    batch_size_train=64,
    batch_size_val=128,
    seed=42,
    shuffle_train=True,
    drop_last_train=False,
    num_workers=0,
    pin_memory=False,
    use_weighted_sampler=False,
    verbose=True,
):
    """
    Like MBSequenceDataset.make_loaders, but works on MBSequenceDatasetTL wrapper.
    Assumes base dataset is already scaled/transformed (fit_and_transform already done elsewhere).
    """
    g = torch.Generator()
    g.manual_seed(seed)

    # Ensure reproducible sampling
    rd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        rd.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_ds = Subset(ds_tl, train_idx)
    val_ds = Subset(ds_tl, val_idx)

    if use_weighted_sampler:
        # weights are based on winter/annual flags stored on the BASE dataset
        base = ds_tl.base
        iw = base.iw[train_idx].cpu().numpy().astype(bool)
        ia = base.ia[train_idx].cpu().numpy().astype(bool)
        n_w, n_a = int(iw.sum()), int(ia.sum())

        if (n_w == 0) or (n_a == 0):
            if verbose:
                print(
                    f"Weighted sampler disabled (one class missing): "
                    f"{n_w} winter | {n_a} annual. Using shuffle instead."
                )
            train_dl = DataLoader(
                train_ds,
                batch_size=batch_size_train,
                shuffle=shuffle_train,
                drop_last=drop_last_train,
                num_workers=num_workers,
                pin_memory=pin_memory,
                worker_init_fn=_seed_worker,
                generator=g,
            )
        else:
            w_w = 1.0
            w_a = n_w / n_a  # >0 since both >0
            sample_weights = np.where(ia, w_a, w_w).astype(np.float32)

            sw_sum = float(sample_weights.sum())
            if (not np.isfinite(sw_sum)) or (sw_sum <= 0.0):
                if verbose:
                    print(
                        "Weighted sampler disabled (invalid weights distribution). "
                        "Using shuffle instead."
                    )
                train_dl = DataLoader(
                    train_ds,
                    batch_size=batch_size_train,
                    shuffle=shuffle_train,
                    drop_last=drop_last_train,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    worker_init_fn=_seed_worker,
                    generator=g,
                )
            else:
                sample_weights = torch.from_numpy(sample_weights)
                sampler = WeightedRandomSampler(
                    sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True,
                    generator=g,
                )
                train_dl = DataLoader(
                    train_ds,
                    batch_size=batch_size_train,
                    sampler=sampler,
                    drop_last=drop_last_train,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    worker_init_fn=_seed_worker,
                    generator=g,
                )
    else:
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size_train,
            shuffle=shuffle_train,
            drop_last=drop_last_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=_seed_worker,
            generator=g,
        )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    if verbose:
        base = ds_tl.base
        n_w_tr, n_a_tr = int(base.iw[train_idx].sum()), int(base.ia[train_idx].sum())
        n_w_va, n_a_va = int(base.iw[val_idx].sum()), int(base.ia[val_idx].sum())
        print(f"Train counts: {n_w_tr} winter | {n_a_tr} annual")
        print(f"Val   counts: {n_w_va} winter | {n_a_va} annual")

    return train_dl, val_dl


def make_finetune_loaders_for_exp(
    cfg,
    tl_assets_for_key,
    batch_size_train=64,
    batch_size_val=128,
    domain_vocab=None,  # optional: {"CH":0,"NOR":1,...}
):
    ds_ft = tl_assets_for_key["ds_finetune"]
    train_idx = tl_assets_for_key["finetune_train_idx"]
    val_idx = tl_assets_for_key["finetune_val_idx"]

    # ---- scaler donor from assets ----
    ds_ch_scalers = tl_assets_for_key["ds_pretrain_scalers"]
    assert (
        ds_ch_scalers.month_mean is not None
    ), "CH scaler donor has no fitted scalers!"

    # ---- source codes aligned with ds_ft.keys ----
    ft_source_codes = tl_assets_for_key.get("ft_source_codes", None)
    if ft_source_codes is None:
        raise KeyError(
            "tl_assets_for_key missing 'ft_source_codes' (required for TL wrapper)."
        )
    if len(ft_source_codes) != len(ds_ft):
        raise ValueError(
            f"ft_source_codes length {len(ft_source_codes)} != len(ds_finetune) {len(ds_ft)}"
        )

    # ---- pick a domain vocab if not explicitly provided ----
    if domain_vocab is None:
        domain_vocab = tl_assets_for_key.get("domain_vocab", None)

    # Optional fallback: build vocab from FT codes only
    if domain_vocab is None:
        uniq = sorted(set(ft_source_codes))
        domain_vocab = {sc: i for i, sc in enumerate(uniq)}

    # ---- clone pristine + apply CH scalers ----
    ds_ft_copy = mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
        ds_ft
    )
    ds_ft_copy.set_scalers_from(ds_ch_scalers)
    ds_ft_copy.transform_inplace()

    # ---- wrap to inject domain labels (SOURCE_CODE / domain_id) ----
    ds_ft_tl = mbm.data_processing.MBSequenceDatasetTL(
        base_ds=ds_ft_copy,
        source_codes=ft_source_codes,
        domain_vocab=domain_vocab,
    )

    # ---- build loaders from wrapper (NOT ds_ft_copy.make_loaders) ----
    ft_train_dl, ft_val_dl = make_loaders_TL(
        ds_ft_tl,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size_train=batch_size_train,
        batch_size_val=batch_size_val,
        seed=cfg.seed,
        shuffle_train=True,
        use_weighted_sampler=True,
        verbose=True,
    )

    return ds_ft_tl, ft_train_dl, ft_val_dl


def make_test_loader_for_key_TL(
    cfg, tl_assets_for_key, batch_size=128, domain_vocab=None
):
    """
    TL-only test loader builder.
    Applies CH scalers to pristine ds_test.
    Returns a TL-wrapped dataset + loader so adapter models can receive domain_id.
    """
    mbm.utils.seed_all(cfg.seed)

    ds_scalers = tl_assets_for_key["ds_pretrain_scalers"]  # fitted CH scaler donor
    ds_test = tl_assets_for_key["ds_test"]  # pristine holdout dataset
    if ds_test is None:
        raise ValueError("TL assets have ds_test=None (no holdout set).")

    # sanity: scalers exist
    if (
        (ds_scalers.month_mean is None)
        or (ds_scalers.static_mean is None)
        or (ds_scalers.y_mean is None)
    ):
        raise ValueError("ds_pretrain_scalers is missing fitted scalers.")

    # clone pristine test
    ds_test_copy = mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
        ds_test
    )

    # Apply CH scalers + transform
    ds_test_copy.set_scalers_from(ds_scalers)
    ds_test_copy.transform_inplace()

    # aligned source codes
    test_source_codes = tl_assets_for_key.get("test_source_codes", None)
    if test_source_codes is None:
        raise KeyError("tl_assets_for_key is missing 'test_source_codes'.")
    if len(test_source_codes) != len(ds_test_copy):
        raise ValueError(
            f"test_source_codes length {len(test_source_codes)} != len(ds_test_copy) {len(ds_test_copy)}"
        )

    # choose vocab (assets-driven preferred)
    if domain_vocab is None:
        domain_vocab = tl_assets_for_key.get("domain_vocab", None)
    if domain_vocab is None:
        uniq = sorted(set(test_source_codes))
        domain_vocab = {sc: i for i, sc in enumerate(uniq)}

    # wrap for TL metadata
    ds_test_tl = mbm.data_processing.MBSequenceDatasetTL(
        base_ds=ds_test_copy,
        source_codes=test_source_codes,
        domain_vocab=domain_vocab,
    )

    # loader (simple, no shuffle)
    test_dl = DataLoader(
        ds_test_tl,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return ds_scalers, ds_test_tl, test_dl, test_source_codes
