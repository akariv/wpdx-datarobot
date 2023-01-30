import datarobot as dr

def get_top_of_leaderboard(project, metric="AUC", feature_list=fl)# , verbose=True):
    """
    A helper method to assemble a dataframe with leaderboard results and print a summary.
    """
    # list of metrics that get better as their value increases
    desc_metric_list = [
        "AUC",
        "Area Under PR Curve",
        "Gini Norm",
        "Kolmogorov-Smirnov",
        "Max MCC",
        "Rate@Top5%",
        "Rate@Top10%",
        "Rate@TopTenth%",
        "R Squared",
        "FVE Gamma",
        "FVE Poisson",
        "FVE Tweedie",
        "Accuracy",
        "Balanced Accuracy",
        "FVE Multinomial",
        "FVE Binomial",
    ]
    asc_flag = False if metric in desc_metric_list else True

    leaderboard = []
    for m in project.get_models():
        leaderboard.append(
            [
                m.blueprint_id,
                m.featurelist.id,
                m.id,
                m.model_type,
                m.sample_pct,
                m.metrics[metric]["validation"],
                m.metrics[metric]["crossValidation"],
            ]
        )
    leaderboard_df = pd.DataFrame(
        columns=[
            "bp_id",
            "featurelist",
            "model_id",
            "model",
            "pct",
            f"validation_{metric}",
            f"cross_validation_{metric}",
        ],
        data=leaderboard,
    )
    leaderboard_top = (
        leaderboard_df[(leaderboard_df["pct"]<84)&(leaderboard_df['featurelist'==fl])] #only get non-frozen models from the final round of autopilot before deployments
        .sort_values(by=f"cross_validation_{metric}", ascending=asc_flag)
        .head()
        .reset_index(drop=True)
    )
    
    return leaderboard_top