import datarobot as dr
import pandas as pd

def get_top_of_leaderboard(project,fl_name=None, metric="AUC"):# , verbose=True):
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
    if fl_name: #enable filtering for certain feature lists
        models=[m for m in project.get_models() if m.featurelist_name==fl_name]
    else:
        models = project.get_models()
        
    for m in models:
        leaderboard.append(
            [
                m.blueprint_id,
                m.featurelist.id,
                m.id,
                m.model_type,
                m.sample_pct,
                m.metrics[metric]["validation"],
                m.metrics[metric]["crossValidation"]
                m.metrics[metric]["holdout"],
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
            f"holdout_{metric}",
        ],
        data=leaderboard,
    )
#     leaderboard_top = (
#         leaderboard_df[(leaderboard_df["pct"]<84)&(leaderboard_df['featurelist']==feature_list.name)] #only get non-frozen models from the final round of autopilot before deployments
#         .sort_values(by=f"cross_validation_{metric}", ascending=asc_flag)
#         .head()
#         .reset_index(drop=True)
#     )
    leaderboard_top = (
        leaderboard_df[round(leaderboard_df["pct"]) <84] #don't use frozen models
        .sort_values(by=f"cross_validation_{metric}", ascending=asc_flag)
        .head()
        .reset_index(drop=True)
    )
    
    return leaderboard_top

def make_cv_data(df, country):
    frame = df.loc[df.clean_country_id==country]

    holdout_rows= round(frame.shape[0]*.15)
    #holdout_fake_date = '1999-01-01'
    print(holdout_rows)
    holdout = frame.sort_values(by='report_date', ascending=False).head(holdout_rows)
    
    indices = holdout.index
    #ug.loc[holdout.index,:]['fold'] = '1999-01-01'
    frame.loc[indices,'fold'] = 'fold_-1'
    #frame['fold_yr']= pd.to_datetime(frame['fold']).apply(lambda x: x.year)
    return frame
