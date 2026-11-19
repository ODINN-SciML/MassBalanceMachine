"""Data of the multi-region (pooled) experiments, shared by notebooks
`3.2 Pooled_model`, `3.3 Pooled_model_pred` and `3.4 Distance`.

- `load_pooled_experiment_data`: stake data, monthly inputs, sub-regions and
  source→target pairs.
- `build_pooled_assets`: training dataset of the multi-region model (all source glaciers).
- `build_ft_assets_id_split`: fine-tuning / test datasets of the target regions
  (measurement-level split, repeated with different seeds).
- `build_zero_shot_test_sets`: full-region test datasets for zero-shot evaluation.
- `count_glaciers_per_region`: number of stake glaciers per region code.
"""

import numpy as np
import pandas as pd
import massbalancemachine as mbm

from papers.Transformer_TF.scripts.config_TF_Europe import (
    RGI_REGIONS,
    IT_GLACIERS,
    AT_GLACIERS,
    VOIS_CLIMATE,
    STATIC_COLS,
    paths_multi,
)
from papers.Transformer_TF.scripts.dataset.data_loader import load_stakes_for_rgi_region
from papers.Transformer_TF.scripts.dataset.xreg_dataset import (
    build_monthly_cache,
    prepare_xreg_pairs_from_cache,
)
from papers.Transformer_TF.scripts.dataset.xreg_lstm_dataset import (
    build_or_load_lstm_dataset_only,
)
from papers.Transformer_TF.scripts.dataset.fd_model import (
    build_assets_from_glacier_list,
)
from papers.Transformer_TF.scripts.utils.glaciers import (
    add_o2region_to_dfs,
    add_o2region_to_monthly_cache,
    split_monthly_cache_by_glaciers,
)


def load_pooled_experiment_data(
    cfg, source_regions, target_regions, target_regions_sub, run_flag_by_code=None
):
    """
    Load the stake data of all regions, their monthly inputs (from
    `WGMS/CrossRegional/monthly_cache/` unless run_flag_by_code says otherwise),
    split Alaska and Central Asia into RGI O2 sub-regions and Italy/Austria by
    country, and assemble one source→(all other regions) pair per source region.

    Returns
    -------
    dict with "dfs", "monthly_cache", "res_xreg_by_source", "split_info_by_source",
    "months_head_pad", "months_tail_pad" (padding of the pooled sequences, from CH).
    """
    # Stake datasets of all regions configured in RGI_REGIONS
    dfs = {rid: load_stakes_for_rgi_region(cfg, rid) for rid in RGI_REGIONS.keys()}

    if run_flag_by_code is None:
        run_flag_by_code = {code: False for code in source_regions + target_regions}

    monthly_cache = build_monthly_cache(
        cfg=cfg,
        dfs=dfs,
        paths_multi=paths_multi,
        vois_climate=VOIS_CLIMATE,
        vois_topographical=STATIC_COLS,
        region_codes=source_regions + target_regions,
        run_flag_by_code=run_flag_by_code,
    )

    # Alaska: RGI O2 sub-regions (glaciers listed under several names are merged)
    dfs, glacier_dict_alaska = add_o2region_to_dfs(
        dfs, "01", cfg, deduplicate_glaciers=True
    )
    monthly_cache = add_o2region_to_monthly_cache(
        monthly_cache, "ALA", glacier_dict_alaska
    )
    monthly_cache = split_monthly_cache_by_glaciers(
        monthly_cache,
        "ALA",
        subregions={
            "ALA_2": {"o2region_col": "O2Region", "values": ["2"]},
            "ALA_4": {"o2region_col": "O2Region", "values": ["4"]},
            "ALA_6": {"o2region_col": "O2Region", "values": ["6"]},
        },
    )

    # Central Asia: RGI O2 sub-regions
    dfs, glacier_dict_ca = add_o2region_to_dfs(dfs, "13", cfg)
    monthly_cache = add_o2region_to_monthly_cache(
        monthly_cache, "CENTRALASIA", glacier_dict_ca
    )
    monthly_cache = split_monthly_cache_by_glaciers(
        monthly_cache,
        "CENTRALASIA",
        subregions={
            "CA_12": {"o2region_col": "O2Region", "values": ["1", "2"]},
            "CA_3": {"o2region_col": "O2Region", "values": ["3"]},
            "CA_4": {"o2region_col": "O2Region", "values": ["4"]},
        },
    )

    # Italy / Austria: split by country
    monthly_cache = split_monthly_cache_by_glaciers(
        monthly_cache,
        "IT_AT",
        subregions={"IT": IT_GLACIERS, "AT": AT_GLACIERS},
        drop_parent=True,
    )

    # One pair per source region: (source, all other regions)
    xreg_pairs = [
        (src, [r for r in source_regions + target_regions_sub if r != src])
        for src in source_regions
    ]
    res_xreg_by_source, split_info_by_source = prepare_xreg_pairs_from_cache(
        cfg=cfg,
        monthly_cache=monthly_cache,
        xreg_pairs=xreg_pairs,
    )

    months_head_pad = res_xreg_by_source["CH"]["months_head_pad"]
    months_tail_pad = res_xreg_by_source["CH"]["months_tail_pad"]
    print(f"months_head_pad: {months_head_pad}")
    print(f"months_tail_pad: {months_tail_pad}")

    return {
        "dfs": dfs,
        "monthly_cache": monthly_cache,
        "res_xreg_by_source": res_xreg_by_source,
        "split_info_by_source": split_info_by_source,
        "months_head_pad": months_head_pad,
        "months_tail_pad": months_tail_pad,
    }


def build_pooled_assets(
    res_xreg_by_source,
    source_regions,
    monthly_cols,
    static_cols,
    cfg,
    months_head_pad,
    months_tail_pad,
    cache_path,
    force_recompute=False,
):
    """
    Training dataset of the multi-region model: all glaciers of the source regions
    (built or loaded from cache_path). Raises an error if any glacier comes from a
    region that is not a source region.
    """
    df_all_glaciers = pd.concat(
        [
            res_xreg_by_source[region]["df_train"][["GLACIER"]]
            .drop_duplicates()
            .assign(region=region)
            for region in source_regions
        ],
        ignore_index=True,
    ).rename(columns={"GLACIER": "glacier"})

    glaciers_all = df_all_glaciers["glacier"].tolist()
    print(f"Total unique glaciers across all source regions: {len(glaciers_all)}")

    assets_full = build_assets_from_glacier_list(
        glaciers=glaciers_all,
        df_ranked=df_all_glaciers,  # glacier -> region lookup
        res_xreg_by_source=res_xreg_by_source,
        monthly_cols=monthly_cols,
        static_cols=static_cols,
        cfg=cfg,
        cache_path=str(cache_path),
        force_recompute=force_recompute,
        months_head_pad=months_head_pad,
        months_tail_pad=months_tail_pad,
        src_regions=source_regions,
    )

    # Failsafe: the pooled training data must only contain source regions
    actual_codes = set(df_all_glaciers.region.unique())
    unexpected = actual_codes - set(source_regions)
    if unexpected:
        raise ValueError(
            f"Pooled training set contains unexpected SOURCE_CODEs: {unexpected}"
        )
    print(f"✓ Pooled set SOURCE_CODEs OK: {actual_codes}")
    print(f"Pooled model assets: {len(assets_full['ds_train'])} sequences")
    return assets_full


def build_ft_assets_id_split(
    monthly_cache,
    regions,
    monthly_cols,
    static_cols,
    cfg,
    cache_dir,
    ft_frac=0.10,
    n_repeats=5,
    force_recompute=False,
    force_recompute_regions=(),
):
    """
    Fine-tuning and test datasets of each target region: the measurement IDs are
    shuffled (winter and annual separately, seed cfg.seed + rep) and the first
    ft_frac of each go to fine-tuning, the rest to the test set. Repeated n_repeats
    times; datasets are cached as `FD_to_<region>_IDsplit_rep<rep>_{FT,TEST}`.

    Returns {region: [{"rep", "seed", "ds_ft", "ds_test", "ft_train_idx",
    "ft_val_idx"}, ...]} (ft_train_idx / ft_val_idx: 80/20 split of the FT set).
    """
    ft_assets_id = {region: [] for region in regions}

    for region in regions:
        print(f"\n{'='*50}\nBuilding FT assets for: {region}")

        df_monthly = monthly_cache[region]["data_monthly"]
        df_monthly_aug = monthly_cache[region]["data_monthly_aug"]
        head_pad = monthly_cache[region]["months_head_pad"]
        tail_pad = monthly_cache[region]["months_tail_pad"]
        force = force_recompute or (region in force_recompute_regions)

        winter_ids_full = (
            df_monthly[df_monthly["PERIOD"] == "winter"]["ID"].unique().tolist()
        )
        annual_ids_full = (
            df_monthly[df_monthly["PERIOD"] == "annual"]["ID"].unique().tolist()
        )

        ft_id_sets = []  # for the check that the repeats differ
        for rep in range(n_repeats):
            winter_ids = list(winter_ids_full)
            annual_ids = list(annual_ids_full)

            rep_seed = cfg.seed + rep
            rng = np.random.default_rng(rep_seed)
            rng.shuffle(winter_ids)
            rng.shuffle(annual_ids)

            n_ft_w = max(1, int(len(winter_ids) * ft_frac)) if winter_ids else 0
            n_ft_a = max(1, int(len(annual_ids) * ft_frac))
            ft_ids = winter_ids[:n_ft_w] + annual_ids[:n_ft_a]
            test_ids = winter_ids[n_ft_w:] + annual_ids[n_ft_a:]
            ft_id_sets.append(set(ft_ids))

            exp_key_id = f"FD_to_{region}_IDsplit_rep{rep}"
            common = dict(
                months_head_pad=head_pad,
                months_tail_pad=tail_pad,
                MONTHLY_COLS=monthly_cols,
                STATIC_COLS=static_cols,
                cache_dir=str(cache_dir),
                force_recompute=force,
                show_progress=False,
            )
            ds_ft_id = build_or_load_lstm_dataset_only(
                cfg=cfg,
                key=f"{exp_key_id}_FT",
                df_loss=df_monthly[df_monthly["ID"].isin(ft_ids)],
                df_full=df_monthly_aug[df_monthly_aug["ID"].isin(ft_ids)],
                kind="ft",
                **common,
            )
            ds_test_id = build_or_load_lstm_dataset_only(
                cfg=cfg,
                key=f"{exp_key_id}_TEST",
                df_loss=df_monthly[df_monthly["ID"].isin(test_ids)],
                df_full=df_monthly_aug[df_monthly_aug["ID"].isin(test_ids)],
                kind="test",
                **common,
            )
            ft_train_idx_id, ft_val_idx_id = (
                mbm.data_processing.MBSequenceDataset.split_indices(
                    len(ds_ft_id), val_ratio=0.2, seed=rep_seed
                )
            )

            ft_assets_id[region].append(
                {
                    "rep": rep,
                    "seed": rep_seed,
                    "ds_ft": ds_ft_id,
                    "ds_test": ds_test_id,
                    "ft_train_idx": ft_train_idx_id,
                    "ft_val_idx": ft_val_idx_id,
                }
            )
            print(
                f"  [ID-split rep {rep}] ds_ft  : {len(ds_ft_id)} seqs | "
                f"ds_test: {len(ds_test_id)} seqs"
            )

        # Check that the repeats use different FT sets
        n_reps = len(ft_id_sets)
        identical_pairs = [
            (i, j)
            for i in range(n_reps)
            for j in range(i + 1, n_reps)
            if ft_id_sets[i] == ft_id_sets[j]
        ]
        if identical_pairs:
            print(
                f"  WARNING [{region}]: identical FT id-sets for repeats {identical_pairs}"
            )
        else:
            overlaps = [
                len(ft_id_sets[i] & ft_id_sets[j]) / len(ft_id_sets[i])
                for i in range(n_reps)
                for j in range(i + 1, n_reps)
            ]
            print(
                f"  [check] {region}: {n_reps} repeats distinct "
                f"(mean overlap {np.mean(overlaps):.1%})"
            )

    return ft_assets_id


def build_zero_shot_test_sets(
    monthly_cache, regions, monthly_cols, static_cols, cfg, cache_dir
):
    """
    Full-region test dataset of each target region (all its stakes), cached as
    `ZEROSHOT_<region>_FULL`. Returns [(region, {"ds_test": dataset}), ...].
    """
    eval_configs = []
    for region in regions:
        ds_test_full = build_or_load_lstm_dataset_only(
            cfg=cfg,
            key=f"ZEROSHOT_{region}_FULL",
            df_loss=monthly_cache[region]["data_monthly"],
            df_full=monthly_cache[region]["data_monthly_aug"],
            months_head_pad=monthly_cache[region]["months_head_pad"],
            months_tail_pad=monthly_cache[region]["months_tail_pad"],
            MONTHLY_COLS=monthly_cols,
            STATIC_COLS=static_cols,
            cache_dir=str(cache_dir),
            force_recompute=False,
            kind="test",
            show_progress=False,
        )
        eval_configs.append((region, {"ds_test": ds_test_full}))
    return eval_configs


def count_glaciers_per_region(dfs):
    """
    Number of stake glaciers per region code, from the stake data (after
    `add_o2region_to_dfs`): ALA, ISL, SJM, NOR, CH, FR, IT, AT and CA_1 to CA_4.
    """
    counts = {
        "ALA": dfs["01"].GLACIER.nunique(),
        "ISL": dfs["06"].GLACIER.nunique(),
        "SJM": dfs["07"].GLACIER.nunique(),
        "NOR": dfs["08"].GLACIER.nunique(),
    }
    df_ceu = dfs["11"].copy()
    df_ceu.loc[df_ceu["GLACIER"].isin(IT_GLACIERS), "SOURCE_CODE"] = "IT"
    df_ceu.loc[df_ceu["GLACIER"].isin(AT_GLACIERS), "SOURCE_CODE"] = "AT"
    for code in ["CH", "FR", "IT", "AT"]:
        counts[code] = df_ceu[df_ceu["SOURCE_CODE"] == code].GLACIER.nunique()
    for o2 in ["1", "2", "3", "4"]:
        counts[f"CA_{o2}"] = dfs["13"][dfs["13"]["O2Region"] == o2].GLACIER.nunique()
    return {k: int(v) for k, v in counts.items()}
