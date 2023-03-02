import math
import dataflows as DF
import datarobot as dr
from pathlib import Path
from wpdx_tools.logger import logger
import hashlib

from wpdx_tools.plus.predict import setup_client


SPLITS = 3
GLOBAL_THRESHOLD = 16000
TIMEOUT = 1800

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


def assign_holdout(country_code_counts):
    def func(rows):
        cur_state = None
        total_rows = 0
        total_holdout = 0
        i = 0
        total_count = 0
        counts_per_country = dict()
        for row in rows:
            cc = row['clean_country_id']
            fold = row['fold']
            state = (fold, cc)
            if cur_state != state:
                cur_state = state
                total_count = country_code_counts[cc]
                counts_per_country.setdefault(cc, 0)
                i = 0
            if i >= total_count * 0.17:
                row['fold'] = -1
                total_holdout += 1
                counts_per_country[cc] += 1
            yield row
            i += 1
            total_rows += 1
        print('HOLDOUT COUNT: {}/{}'.format(total_holdout, total_rows))
        for cc, c in counts_per_country.items():
            print('- COUNTRY {}: {}/{}'.format(cc, c, country_code_counts[cc]))
    return func


def hasher(val):
    return hashlib.sha1(val.encode('utf-8')).hexdigest()[:8]


def apply_type_transforms(project: dr.Project, fl: dr.Featurelist):
    VAR_TYPES = dict(
        (f, 'categoricalInt') for f in ['isobioclimates', 'landcover_code']
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


def get_top_of_leaderboard(project: dr.Project, feature_list=None):
    metric = 'AUC'
    reverse = True
    print('GET TOP OF LEADERBOARD ({}, {}):'.format(project.id, feature_list))
    leaderboard = []
    for m in project.get_models():
        if m.sample_pct < 85 and feature_list in (m.featurelist.id, None):
            if not m.metrics[metric]["crossValidation"]:
                try:
                    mj = m.cross_validate()
                    m = mj.get_result_when_complete(max_wait=3600*24)
                    m = dr.Model.get(m.project_id, m.id)
                except Exception as e:
                    print('CROSS VALIDATION FAILED: {}'.format(e))
            if m.metrics[metric]["crossValidation"]:
                leaderboard.append((m.metrics[metric]["crossValidation"], m.id))
        print('\t{}: pct: {}, fl: {}, AUC metrics: {!r}, frozen: {!r}'.format(
            m.id, m.sample_pct, m.featurelist.id, m.metrics[metric], m.is_frozen
        ))
        if len(leaderboard) == 10:
            break
    
    assert len(leaderboard) > 0, 'No models found for project {} and fl {}'.format(project.id, feature_list)

    return sorted(leaderboard, reverse=reverse)[0][1]

def do_predictions(country_codes, training_path, base_path):
    TMPDIR = Path(base_path) / 'intermediate'

    if not setup_client():
        return

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

    country_codes = list(country_codes)
    country_code_counts = dict((c, sum(v for k, v in fold_counts.items() if c == k[0])) for c in country_codes)

    prediction_server: dr.PredictionServer = None
    try:
        prediction_server = dr.PredictionServer.list()[0]
        print('PREDICTION SERVER: {}'.format(prediction_server))
    except IndexError:
        print('No prediction server found')

    # TRAINING THE MODEL
    for country_code in ['all'] + country_codes:
        print(f'TRAINING FOR {country_code}')
        outdir = TMPDIR / country_code

        # total_count = sum(v for k, v in fold_counts.items() if country_code in ('all', k[0]))

        training_outfile = f'data/{country_code}_training_water_points.csv'
        dp, _ = DF.Flow(
            DF.load(training_path),
            # DF.checkpoint('prediction'),
            DF.filter_rows(lambda r: country_code in ('all', r['clean_country_id'])),
            DF.add_field('mangled_id', 'string', lambda r: hasher(r['wpdx_id'])),
            DF.sort_rows('{clean_country_id}{clean_adm1}{clean_adm2}{mangled_id}'),
            assign_folds(fold_counts),
            # DF.printer(),
            DF.sort_rows('{fold}{clean_country_id}{report_date}'),
            assign_holdout(country_code_counts),
            DF.delete_fields(['lat_deg', 'lon_deg', 'mangled_id']),
            DF.update_resource(-1, path=training_outfile),
            DF.dump_to_path(outdir / 'training'),
        ).process()
        fields = [f.name for f in dp.resources[0].schema.fields]
        fields = [f for f in fields if f not in ('fold', 'wpdx_id', 'report_date', 'clean_country_id', 'clean_adm1', 'clean_adm2', 'unified_install_year', 'unified_installer')]

        project_name = f'Status Predictions - {country_code}'

        deployment: dr.Deployment = [x for x in dr.Deployment.list() if x.label == project_name]
        if len(deployment) > 0:
            deployment = deployment[0]
        else:
            deployment = None

        project: dr.Project = None
        model: dr.Model = None
        existing_projects = dr.Project.list(search_params=dict(project_name=project_name))
        if len(existing_projects) > 0:
            for project in existing_projects:
                print('FOUND EXISTING PROJECT, DELETING: {}: {}'.format(project.id, project))
                try:
                    project.delete()
                except Exception as e:
                    print('FAILED TO DELETE PROJECT: {}'.format(e))
            project = None

        if project is None:
            try:
                print(f'CREATE DATASET: {country_code}')
                dataset = dr.Dataset.create_from_file(file_path=outdir / 'training' / training_outfile)
                print(f'CREATE PROJECT: {country_code}')
                project: dr.Project = dr.Project.create_from_dataset(dataset.id, project_name=f'Status Predictions - {country_code}')
                print(f'CREATE FL: {country_code}')
                feature_list = project.create_featurelist(name=f'FL-{country_code}', features=fields)
                feature_list = apply_type_transforms(project, feature_list)

                advanced_options = dr.helpers.AdvancedOptions(
                    protected_features=['clean_adm1'] if country_code != 'all' else ['clean_country_id'],
                    preferable_target_value='No',
                    fairness_metrics_set='favorableAndUnfavorablePredictiveValueParity',
                    fairness_threshold='0.8'
                )
                print(f'ANALYZE AND MODEL: {country_code}')
                project.analyze_and_model(
                    target='status_id',
                    mode=dr.enums.AUTOPILOT_MODE.FULL_AUTO,
                    advanced_options=advanced_options,
                    partitioning_method=dr.UserCV('fold', -1, seed=0),
                    featurelist_id=feature_list.id,
                    worker_count=-1,
                )
                project.wait_for_autopilot()
                print(f'ANALYZE DONE: {country_code}')
                leaderboard_top = get_top_of_leaderboard(project)
                m: dr.Model = dr.Model.get(project=project.id, model_id=leaderboard_top)

                ret = project.unlock_holdout()
                print('UNLOCK HOLDOUT DONE', ret)

                candidates = []
                top_features = m.get_or_request_feature_impact()
                while len(top_features) > (35 if country_code == 'all' else 20):
                    print('REMOVING FEATURES, CURRENTLY {} FEATURES: {}...{}'.format(len(top_features), top_features[0], top_features[-1]))
                    n = len(top_features) - 1
                    top_features = top_features[:n]
                    new_fl = project.create_featurelist(feature_list.name + f'-{n}', [f['featureName'] for f in top_features])
                    mj = m.retrain(featurelist_id = new_fl.id)
                    m = mj.get_result_when_complete(max_wait=3600*24)
                    m = dr.Model.get(m.project_id, m.id)
                    mj = m.cross_validate()
                    m = mj.get_result_when_complete(max_wait=3600*24)
                    m = dr.Model.get(m.project_id, m.id)
                    if m.metrics['AUC']['holdout'] and m.metrics['AUC']['crossValidation']:
                        candidates.append((m.id, new_fl.id, n, m.metrics['AUC']['holdout'], m.metrics['AUC']['crossValidation']))
                    else:
                        print('SKIPPING MODEL, MISSING METRICS: {}'.format(m.metrics))

                    top_features = m.get_or_request_feature_impact()
                feature_list = new_fl
                print('FINAL FEATURES: {}'.format(top_features))
                assert len(candidates) > 0
                print('NUM OF CANDIDATES: {}'.format(len(candidates)))

                max_m1 = max(candidates, key=lambda c: c[3])
                max_m2 = max(candidates, key=lambda c: c[4])
                fl_candidates = [c for c in candidates if c[3] >= max_m1[3]*0.95 and c[4] >= max_m2[4]*0.95]
                top_fl_candidate = min(fl_candidates, key=lambda c: c[2])[1]
                print('TOP FEATURE LIST: {}'.format(top_fl_candidate))

                project.start_autopilot(top_fl_candidate, mode=dr.enums.AUTOPILOT_MODE.COMPREHENSIVE)
                project.wait_for_autopilot()
                print(f'AUTOPILOT DONE {country_code}')

                leaderboard_top = get_top_of_leaderboard(project, feature_list=top_fl_candidate)
                model: dr.Model = dr.Model.get(project=project.id, model_id=leaderboard_top)
                project.start_prepare_model_for_deployment(model_id=model.id)
                print('MODEL: {}: {}'.format(model.id, model))
            except Exception as e:
                print(f'ERROR PREDICTING FOR {country_code}: {e}')
                logger.exception(e)
                continue

        if model is not None and prediction_server is not None:
            print('DEPLOYING MODEL: {}'.format(model.id))
            if deployment is None:
                d = dr.Deployment.create_from_learning_model(model.id, label=project_name, default_prediction_server_id=prediction_server.id)
                print('NEW DEPLOYMENT: {}: {}'.format(d.id, d))
            else:
                print('EXISTING DEPLOYMENT: {}: {}'.format(deployment.id, deployment))
                deployment.replace_model(model.id, 'SCHEDULED_REFRESH', max_wait=TIMEOUT)


