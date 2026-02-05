from scripts.config_FR import *


def getDfAggregatePred(test_set, y_pred_agg, all_columns):
    # Aggregate predictions to annual or winter:
    df_pred = test_set["df_X"][all_columns].copy()
    df_pred["target"] = test_set["y"]
    grouped_ids = df_pred.groupby("ID").agg(
        {"target": "mean", "YEAR": "first", "POINT_ID": "first", "GLACIER": "first"}
    )
    grouped_ids["pred"] = y_pred_agg
    grouped_ids["PERIOD"] = (
        test_set["df_X"][all_columns].groupby("ID")["PERIOD"].first()
    )

    return grouped_ids
