import numpy as np
import random as rd
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

import massbalancemachine as mbm


def make_dan_loaders_for_exp(
    cfg,
    tl_assets_for_key,
    *,
    batch_size_train=64,
    batch_size_val=128,
    seed=42,
    shuffle_train=True,
    drop_last_train=False,
    num_workers=0,
    pin_memory=False,
    # If FT is tiny, upsample it so CH doesn’t dominate
    mix_ratio_ft=1.0,
    domain_vocab=None,
    verbose=True,
):
    """
    DAN loaders:
      Train: mixed CH_train + FT_train
      Val:   FT_val only (task performance where you care)

    Returns:
      domain_vocab, train_dl, val_dl
    """
    g = torch.Generator()
    g.manual_seed(seed)

    rd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        rd.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # ---- assets ----
    ds_ch_pristine = tl_assets_for_key["ds_pretrain"]
    ds_ft_pristine = tl_assets_for_key["ds_finetune"]
    ds_scalers = tl_assets_for_key["ds_pretrain_scalers"]

    ch_train_idx = tl_assets_for_key["pretrain_train_idx"]
    ft_train_idx = tl_assets_for_key["finetune_train_idx"]
    ft_val_idx = tl_assets_for_key["finetune_val_idx"]

    ch_source_codes = tl_assets_for_key["pretrain_source_codes"]
    ft_source_codes = tl_assets_for_key["ft_source_codes"]

    if domain_vocab is None:
        domain_vocab = tl_assets_for_key.get("domain_vocab", None)
    if domain_vocab is None:
        uniq = sorted(set(ch_source_codes) | set(ft_source_codes))
        domain_vocab = {sc: i for i, sc in enumerate(uniq)}

    # ---- clone pristine + apply CH scalers ----
    ds_ch = mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
        ds_ch_pristine
    )
    ds_ch.set_scalers_from(ds_scalers)
    ds_ch.transform_inplace()

    ds_ft = mbm.data_processing.MBSequenceDataset._clone_untransformed_dataset(
        ds_ft_pristine
    )
    ds_ft.set_scalers_from(ds_scalers)
    ds_ft.transform_inplace()

    # ---- wrap TL metadata (domain_id) ----
    ds_ch_tl = mbm.data_processing.MBSequenceDatasetTL(
        base_ds=ds_ch,
        source_codes=ch_source_codes,
        domain_vocab=domain_vocab,
    )
    ds_ft_tl = mbm.data_processing.MBSequenceDatasetTL(
        base_ds=ds_ft,
        source_codes=ft_source_codes,
        domain_vocab=domain_vocab,
    )

    # ---- subsets ----
    ch_train = Subset(ds_ch_tl, ch_train_idx)
    ft_train = Subset(ds_ft_tl, ft_train_idx)
    ft_val = Subset(ds_ft_tl, ft_val_idx)

    # ---- mix train ----
    if mix_ratio_ft is None or mix_ratio_ft <= 0:
        train_mix = ConcatDataset([ch_train, ft_train])
    else:
        n_ch = len(ch_train)
        n_ft = max(len(ft_train), 1)
        k = int(np.ceil((n_ch * float(mix_ratio_ft)) / n_ft))
        k = max(k, 1)
        train_mix = ConcatDataset([ch_train] + [ft_train] * k)

    train_dl = DataLoader(
        train_mix,
        batch_size=batch_size_train,
        shuffle=shuffle_train,
        drop_last=drop_last_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    val_dl = DataLoader(
        ft_val,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    if verbose:
        print(f"[DAN] domain_vocab size={len(domain_vocab)}")
        print(
            f"[DAN] CH train={len(ch_train)} | FT train={len(ft_train)} | FT val={len(ft_val)} | mix={len(train_mix)}"
        )
        b = next(iter(train_dl))
        print(f"[DAN] batch domain_id unique: {torch.unique(b['domain_id']).tolist()}")

    return domain_vocab, train_dl, val_dl
