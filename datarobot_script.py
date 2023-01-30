import math
import dataflows as DF
import datarobot as dr
from pathlib import Path
import datetime
import pandas
from wpdx_tools.logger import logger
import hashlib
from helpers import get_top_of_leaderboard


DATAROBOT_API_KEY = 'xxx'
SPLITS = 3
GLOBAL_THRESHOLD = 16000
TIMEOUT = 1800
reuse = False
CATEGORIES = {
    (True, True, True): 'ok',
    (True, True, False): 'maintenance',
    (True, False, True): 'maintenance',
    (True, False, False): 'non-func-new',
    (False, True, True): 'func-new',
    (False, True, False): 'maintenance',
    (False, False, True): 'maintenance',
    (False, False, False): 'repair',
}

def ts(date):
    ret = datetime.datetime.combine(date, datetime.time(0, 0, 0, 0))
    return ret


def update_dates(year_diff):
    now = ts(datetime.datetime.now() + datetime.timedelta(days=365.2425*year_diff))
    print(f'CALCULATED REPORT DATE: +{year_diff} = {now}')

    def func(row):
        install_year = row.get('unified_install_year')
        if install_year:
            install_year = int(install_year)
            age = (now - datetime.datetime(install_year, 1, 1)).days / 365.2425
            row['age_in_years'] = age

        last_report_date = ts(row['report_date'])
        row['years_since_report'] = (now - last_report_date).days / 365.2425
        # row['report_date'] = report_date
    return DF.Flow(
        DF.add_field('years_since_report', 'number'),
        func,
        DF.add_field('orig_status_id', 'string', lambda r: r['status_id']),
        DF.delete_fields(['report_date', 'status_id'])
    )


def assign_folds(fold_counts):
    count = 0
    last_key = None

    def func(row):
        nonlocal count
        nonlocal last_key
        key = (row['clean_country_id'], row['clean_adm1'], row['clean_adm2'])
        if key != last_key:
            count = 0
            last_key = key
        pct = count / fold_counts[key]
        count += 1
        fold_idx = math.floor(5 * pct)
        row['fold'] = fold_idx
        return row
    return DF.Flow(
        DF.add_field('fold', 'integer', None),
        func
    )


def assign_holdout(total_count):  #since this is applied after sorting, looks like it works
    def func(rows):
        for i, row in enumerate(rows):
            if i >= total_count * 0.85:
                row['fold'] = -1
            yield row
    return func


def hasher(val):
    return hashlib.sha1(val.encode('utf-8')).hexdigest()[:8]


def apply_type_transforms(project: dr.Project, fl: dr.Featurelist):
    VAR_TYPES = dict(
        (f, 'categoricalInt') for f in ['acled_index', 'isobioclimates', 'landcover_code']
    )
    return project.create_featurelist(
        name=fl.name + '-typed',
        features=[
            project.create_type_transform_feature(
                name=f + '-categorical',
                parent_name=f,
                variable_type=VAR_TYPES[f]
            ).name
            if f in VAR_TYPES
            else f 
            for f in fl.features]
    )
    

def do_predictions(country_codes, training_path, prediction_path, validation_path, base_path, years_range):
    output_path = base_path + 'prediction_map/'
    TMPDIR = Path(base_path) / 'intermediate'

    dr.Client(token=DATAROBOT_API_KEY, endpoint='https://app.datarobot.com/api/v2')

    fold_counts = DF.Flow(
        DF.load(training_path),
        # DF.checkpoint('prediction'),
        DF.update_resource(0, name='prediction'),
        DF.select_fields(['clean_country_id', 'clean_adm1', 'clean_adm2']),
        DF.join_with_self('prediction', ['clean_country_id', 'clean_adm1', 'clean_adm2'], dict(
            clean_country_id=None,
            clean_adm1=None,
            clean_adm2=None,
            count=dict(aggregate='count')
        )),
        # DF.printer(),
    ).results()[0][0]
    fold_counts = dict(((r['clean_country_id'], r['clean_adm1'], r['clean_adm2']), r['count']) for r in fold_counts)
    total_count = sum(fold_counts.values())

    prediction_map = dict()
    projects = dict()
    models = dict()
    datasets = dict()
    country_codes = list(country_codes)
    saved_wpdx_ids = dict()
    saved_status_ids = dict()
    validation_scores = []
    prediction_prep = dict()

    # TRAINING THE MODEL
    for country_code in ['all'] + country_codes:
        print(f'TRAINING FOR {country_code}')
        outdir = TMPDIR / country_code

        training_outfile = f'data/{country_code}_training_water_points.csv'
        dp, _ = DF.Flow(
            DF.load(training_path),
            # DF.checkpoint('prediction'),
            DF.filter_rows(lambda r: country_code in ('all', r['clean_country_id'])),
            DF.add_field('mangled_id', 'string', lambda r: hasher(r['wpdx_id'])),
            DF.sort_rows('{clean_adm1}{clean_adm2}{mangled_id}'),
            assign_folds(fold_counts),
            # DF.printer(),
            DF.sort_rows('{report_date}'),
            assign_holdout(total_count),  #this becomes the validation fold in the Validation section below 
            DF.delete_fields(['lat_deg', 'lon_deg', 'mangled_id']),
            DF.update_resource(-1, path=training_outfile),
            DF.dump_to_path(outdir / 'training'),
        ).process()
        fields = [f.name for f in dp.resources[0].schema.fields]
        fields = [f for f in fields if f not in ('fold', 'wpdx_id', 'report_date', 'clean_country_id', 'clean_adm1', 'clean_adm2')] #unified_install_year
        
        prediction_outfile = f'data/{country_code}_prediction_water_points.csv'
        rows = DF.Flow(
            DF.load(prediction_path),
            DF.filter_rows(lambda r: country_code in ('all', r['clean_country_id'])),
            DF.add_field('fold', 'date', None),
            DF.update_resource(-1, path=prediction_outfile),
            DF.dump_to_path(outdir / 'prediction'),
            DF.select_fields(['wpdx_id', 'status_id']),
        ).results()[0][0]

        # rps = sorted([r['report_date'] for r in rows])
        wpdx_ids = [r['wpdx_id'] for r in rows]
        status_ids = [r['status_id'] for r in rows]
        if len(wpdx_ids) < 100:
            #Is this used to apply the country model? If so, I think the threshold was to use the global model if < 16000 WPDx IDs per country
            print(f'SKIPPING PREDICTION FOR {country_code}, only {len(wpdx_ids)} water points')
            continue
        if country_code != 'all':
            saved_wpdx_ids[country_code] = wpdx_ids
            saved_status_ids[country_code] = status_ids

        project_name = f'Status Predictions - {country_code}'
        project = None
        existing_projects = dr.Project.list(search_params=dict(project_name=project_name))
        if len(existing_projects) > 0:
            project = existing_projects[0]
            if not reuse: #not sure on logic here
                print('FOUND EXISTING PROJECT, DELETING: {}'.format(project.project_name)) #not sure this is necessary
                project.delete()
                project = None
            else:
                print('FOUND EXISTING PROJECT, NOT DELETING: {}'.format(project.project_name))
                projects[country_code] = project
                try:
                    model: dr.Model = dr.ModelRecommendation.get(project.id).get_model()
                    models[country_code] = model
                except Exception as e:
                    print(f'FAILED TO GET MODEL for {country_code}: {e}')
                    continue

        if project is None:
            try:
                dataset = dr.Dataset.create_from_file(file_path=outdir / 'training' / training_outfile)
                datasets[country_code] = dataset
                
                project: dr.Project = dr.Project.create_from_dataset(dataset.id, project_name=f'Status Predictions - {country_code}')
                feature_list = project.create_featurelist(name=f'FL-{country_code}', features=fields)
                feature_list = apply_type_transforms(project, feature_list)
                projects[country_code] = project

                advanced_options = dr.helpers.AdvancedOptions(
                    protected_features=['clean_adm1'] if country_code != 'all' else ['clean_country_id'], #nice
                    preferable_target_value='No',
                    fairness_metrics_set='favorableAndUnfavorablePredictiveValueParity',
                    fairness_threshold='0.8'
                )
                project.analyze_and_model(
                    target='status_id',
                    mode=dr.enums.AUTOPILOT_MODE.QUICK, #update to full auto
                    advanced_options=advanced_options,
                    partitioning_method=dr.UserCV('fold', -1, seed=0),
                    featurelist_id=feature_list.id,
                    worker_count=-1,
                )
                project.wait_for_autopilot()
                print('ANALYZE DONE')
    
                leaderboard_top = get_top_of_leaderboard(project, metric="AUC", feature_list=feature_list.name)
                m: dr.Model = project.get(project=project.id, model_id=leaderboard_top.iloc[0]["model_id"] #gets the best 
                #m: dr.Model = project.get_top_model() #this is going to be a frozen model (trained up to 100%, ready to deploy). 
                
                top_features = m.get_or_request_feature_impact()
                while len(top_features) > 10:  # update stopping criteria
                    print('REMOVING FEATURES, CURRENTLY {} FEATURES: {}...{}'.format(len(top_features), top_features[0], top_features[-1]))
                    n = len(top_features) - 1
                    top_features = top_features[:n]
                    new_fl = project.create_featurelist(feature_list.name + f'-{n}', [f['featureName'] for f in top_features])
                    mj = m.retrain(featurelist_id = new_fl.id)
                    m = mj.get_result_when_complete()
                    m = dr.Model.get(m.project_id, m.id)
                    m.cross_validate()
                    top_features = m.get_or_request_feature_impact()
                feature_list = new_fl
                print('FINAL FEATURES: {}'.format(top_features))

#                 advanced_options.prepare_model_for_deployment = True
#                 project.analyze_and_model(
#                     target='status_id',
#                     mode=dr.enums.AUTOPILOT_MODE.COMPREHENSIVE,
#                     advanced_options=advanced_options,
#                     partitioning_method=dr.UserCV('fold', -1, seed=0),
#                     featurelist_id=feature_list.id,
#                     worker_count=-1,
#                 )
                project.start_autopilot(feature_list.id) # start comprehensive mode
                project.wait_for_autopilot()
                print('AUTOPILOT DONE')

                #model: dr.Model = dr.ModelRecommendation.get(project.id).get_model()
                leaderboard_top = get_top_of_leaderboard(project, metric="AUC", feature_list=feature_list.name) #get best model from reduced feature list,name space is repeated in the loop..
                model: dr.Model = project.get(project=project.id, model_id=leaderboard_top.iloc[0]["model_id"] #gets the best model from the new feature list 
                                          
                project.start_prepare_model_for_deployment(model_id=model.id)
                models[country_code] = model
                print('MODEL: {}'.format(model))
            except Exception as e:
                print(f'ERROR PREDICTING FOR {country_code}: {e}')
                logger.exception(e)
                continue

    # VALIDATING THE MODELS-------------> I'm not sure this is necessary since you are using the holdout fold. If you do project.unlock_holdout() we should be able to see these metrics
    for country_code in country_codes:
        print(f'VALIDATING FOR {country_code}')
        outdir = TMPDIR / country_code
        validation_file = f'data/{country_code}_water_points.csv'  #this should be achieved with the holdout fold now
        validation_dir = outdir / f'validation'

        expected = DF.Flow(
            DF.load(str(outdir / 'training' / 'datapackage.json')),
            DF.rename_fields(dict(status_id='expected_status_id')),
            DF.update_resource(-1, path=validation_file),
            DF.dump_to_path(validation_dir),
            DF.select_fields(['expected_status_id']),
        ).results()[0][0]

        selected_model_idx = 'all'
        if len(expected) > GLOBAL_THRESHOLD:
            selected_model_idx = country_code

        for model_idx in ['all', country_code]:
            try:
                project: dr.Project = projects[model_idx]
                model = models[model_idx]
            except KeyError:
                continue

            validation_dataset = project.upload_dataset(str(validation_dir / validation_file), max_wait=TIMEOUT) #this should be achieved with the holdout fold now
            print(validation_dataset) #
            # validation_dataset.wait_for_upload()

            try: # you should get all of this for free from the holdout fold
                job: dr.PredictJob = model.request_predictions(validation_dataset.id)
                print('VALIDATION JOB: {}'.format(job))
                job.wait_for_completion()
                print('VALIDATION JOB DONE')
                predictions: pandas.DataFrame = dr.PredictJob.get_predictions(project.id, job.id)
                comparison = [
                    dict(**r, predicted=p['class_Yes'])
                    for r, p in zip(expected, predictions.to_dict('records'))
                ]
                assert len(comparison) == len(expected)
                score = 0
                threshold_score = None
                for threshold in range(0, 100):
                    threshold = threshold / 100
                    actual = [(1 if r['expected_status_id'] == 'Yes' else 0) for r in comparison]
                    predicted = [(1 if r['predicted'] >= threshold else 0) for r in comparison]
                    tp = sum([1 for a, p in zip(actual, predicted) if a == 1 and p == 1])
                    fp = sum([1 for a, p in zip(actual, predicted) if a == 0 and p == 1])
                    tn = sum([1 for a, p in zip(actual, predicted) if a == 0 and p == 0])
                    fn = sum([1 for a, p in zip(actual, predicted) if a == 1 and p == 0])
                    precision = tp / (tp + fp) if tp + fp > 0 else 0
                    recall = tp / (tp + fn) if tp + fn > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                    if f1 > score:
                        score = f1
                        threshold_score = threshold
                
                validation_scores.append(
                    dict(
                        country_code=country_code,
                        global_model=model_idx == 'all', 
                        score=score,
                        threshold_score=threshold_score,
                        count=len(comparison))
                    )
                print('VALIDATION SCORE', validation_scores[-1])
                if model_idx == selected_model_idx:
                    prediction_prep[country_code] = dict(
                        project=project,
                        model=model,
                        threshold=threshold_score,
                    )

            except Exception as e:
                print(f'ERROR VALIDATING FOR {model_idx}, {country_code}: {e}')
            
    DF.Flow(
        validation_scores,
        DF.printer(num_rows=50),
        DF.dump_to_path(output_path + '/validation/')
    ).process()

    # DOING THE PREDICTIONS
    for country_code in country_codes:
        print(f'PREDICTING FOR {country_code}')
        outdir = TMPDIR / country_code

        if country_code not in prediction_prep:
            print('SKIPPING PREDICTION FOR {}, NO MODEL SELECTED'.format(country_code))
            continue
        prep_rec = prediction_prep[country_code]
        project = prep_rec['project']
        model = prep_rec['model'] #
        threshold = prep_rec['threshold']
        wpdx_ids = saved_wpdx_ids[country_code]
        status_ids = saved_status_ids[country_code]

        for years in years_range:
            pred_file = f'data/{country_code}_water_points_{years}y.csv'
            pred_dir = outdir / f'{years}y'
            DF.Flow(
                DF.load(str(outdir / 'prediction' / 'datapackage.json')),
                update_dates(years),
                DF.update_resource(-1, path=pred_file),
                DF.dump_to_path(pred_dir),
            ).process()

            pred_dataset = project.upload_dataset(str(pred_dir / pred_file), max_wait=TIMEOUT)
            job: dr.PredictJob = model.request_predictions(pred_dataset.id)
            print('PREDICTION JOB: {}'.format(job))
            try:
                job.wait_for_completion()
            except Exception as e:
                print(f'ERROR PREDICTING FOR {country_code}, {years}y: {e}')
                continue
            print('PREDICTION JOB DONE')
            predictions: pandas.DataFrame = dr.PredictJob.get_predictions(project.id, job.id)
            # print('PREDICTIONS: {}'.format(predictions))
            predictions = predictions.to_dict('records')
            assert len(predictions) == len(wpdx_ids)
            predictions = [
                {
                    'wpdx_id': wpdx_id,
                    'status_id': status_id,
                    f'prediction_yes_{years}y': pred['class_Yes'],
                    f'prediction_no_{years}y': pred['class_No'],
                    f'predicted_status_{years}y': 'Yes' if pred['class_Yes'] >= threshold else 'No',
                } for wpdx_id, status_id, pred in zip(wpdx_ids, status_ids, predictions)
            ]

            # prediction_explanations = None
            # if years == years_range[0]:
            #     # Run feature impact
            #     try:
            #         fi_job = model.request_feature_impact()
            #         print(f'Awaiting Feature Impact for {country_code}')
            #         fi_job.wait_for_completion()
            #     except dr.errors.JobAlreadyRequested:
            #         print('Feature Impact already requested')
            #         fi_job = None

            #     # Run Prediction explanations
            #     pei_job = None
            #     try:
            #         print(f'Checking PE Initialization for {country_code}')
            #         pei = dr.PredictionExplanationsInitialization.get(project.id, model.id)
            #     except dr.errors.ClientError:
            #         print(f'Requesting PE Initialization for {country_code}')
            #         pei_job = dr.PredictionExplanationsInitialization.create(project.id, model.id)
            #     if pei_job is not None:
            #         print(f'Awaiting PE Initialization for {country_code}')
            #         pei_job.wait_for_completion()

            #     print(f'Requesting Prediction Explanations for {country_code}')
            #     pe_job = dr.PredictionExplanations.create(project.id, model.id, pred_dataset.id, max_explanations=7)
            #     print(f'waiting...', pe_job, pe_job.id, pe_job.status)
            #     pe_job.wait_for_completion(max_wait=3600)
            #     print(f'done', pe_job, pe_job.id, pe_job.status)
            #     pe = pe_job.get_result()
            #     print(f'getting rows...')
            #     prediction_explanations = [dict(
            #         prediction_explanations=[dict(
            #             l=r['label'],
            #             s=int(r['strength'] * 100),
            #             q=r['qualitative_strength'],
            #             f=r['feature'],
            #         ) for r in rr.prediction_explanations]
            #     ) for rr in pe.get_rows()]
            #     print(f'got {len(prediction_explanations)} rows:', prediction_explanations[:3])


            # if years == 0:
            #     nos = [p for p in predictions if p['predicted_status_0y'] == 'No']
            #     for p in predictions:
            #         p['predicted_high_risk'] = False
            #     nos = sorted(nos, key=lambda p: p['prediction_no_0y'], reverse=True)
            #     nos = nos[:len(nos) // 10]
            #     for p in nos:
            #         p['predicted_high_risk'] = True

            for p in predictions:
                prediction_map.setdefault(p['wpdx_id'], dict()).update(p)
            # if prediction_explanations is not None:
            #     for wpdx_id, pe in zip(wpdx_ids, prediction_explanations):
            #         prediction_map.setdefault(wpdx_id, dict()).update(pe)
        
            try:
                dr.Dataset.delete(pred_dataset.id)
            except Exception as e:
                print(f'ERROR DELETING PREDICTION DATASET FOR {country_code}, {years}', e) 

        for point in prediction_map.values():
            values = (
                point['status_id'] == 'Yes',
                point['predicted_status_0y'] == 'Yes',
                point['predicted_status_2y'] == 'Yes'
            )
            point['predicted_category'] = CATEGORIES[values]

        # project.delete()
        try:
            dataset = datasets[country_code]
            dr.Dataset.delete(dataset.id)
        except Exception as e:
            print(f'ERROR DELETING DATASET FOR {country_code}', e) 

    print('PREDICTIONS SAVED')
    DF.Flow(
        prediction_map.values(),
        DF.dump_to_path(output_path)
    ).process()
