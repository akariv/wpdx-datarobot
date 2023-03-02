from pathlib import Path

import pandas
import datetime

import dataflows as DF
import datarobot as dr

from wpdx_tools.plus.params import get_param
from wpdx_tools.plus.prediction_fields import calc_age
from wpdx_tools.plus.prediction_fields import PREDICTION_ADDED_FIELDS
from wpdx_tools.prediction_config import YEARS_RANGE

FIELDS = [
    'wpdx_id',
    'status_id',

    'clean_country_id',
    'clean_adm1',
    'clean_adm2',
    'lat_deg',
    'lon_deg',

    'water_tech_clean',
    'water_source_clean',
    'water_tech_category',
    'water_source_category',
    'management_clean',
    'pay_clean',
    'subjective_quality_clean',
    # 'source',

    'report_date',
    'unified_install_year',
    # 'unified_installer',
    'age_in_years',

    'distance_to_primary',
    'distance_to_secondary',
    'distance_to_tertiary',
    'distance_to_city',
    'distance_to_town',

    'is_urban',

    'precipitation_5year',
    'precipitation_10year',
    'landcover_code',
    'acled_index',
    'isobioclimates',

    'bgs_dtw',
    'bgs_prod',
    'bgs_stor',
    'bgs_recharge',

    'water_risk',

    'rwi',

    'assigned_population',
    'local_population',
    'usage_cap',
    'pressure',
    'crucialness',
]
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
BASE_PATH = '/app/site/static/'
TMPDIR = Path(BASE_PATH) / 'intermediate'
TIMEOUT = 60 * 60 * 24
OUTPUT_PATH = BASE_PATH + 'prediction_map/'

def update_country_codes(cc):
    def func(row):
        cc.add(row.get('clean_country_id'))
    return func


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


def setup_client():
    DATAROBOT_API_KEY = get_param('datarobot-api-key')
    if not DATAROBOT_API_KEY:
        print('NO DATAROBOT API KEY, SKIPPING PREDICTION')
        return False
    dr.Client(token=DATAROBOT_API_KEY, endpoint='https://app.datarobot.com/api/v2')
    return True


def get_deployment(country_code):
    project_name = f'Status Predictions - {country_code}'
    deployment = None
    deployments = dr.Deployment.list(search=project_name)
    for d in deployments:
        if d.label == project_name:
            deployment = d
            break
    return deployment


def predict(checkpoint):

    if not setup_client():
        return
    
    fields = FIELDS + PREDICTION_ADDED_FIELDS

    country_codes = set()
    DF.Flow(
        DF.checkpoint(checkpoint),
        DF.filter_rows(lambda r: r.get('status_clean') not in (
            'Abandoned/Decommissioned',
        )),
        DF.select_fields(fields + ['history']),
        # DF.filter_rows(lambda r: r['clean_country_id'] in ('ETH', 'UGA', 'SLE', 'GHA')), # TODO - remove
        update_country_codes(country_codes),
        DF.dump_to_path(BASE_PATH + 'prediction-predict'),
    ).process()

    prediction_map = dict()

    # DOING THE PREDICTIONS
    for country_code in country_codes:
        print(f'PREDICTING FOR {country_code}')

        outdir = TMPDIR / country_code
        deployment = get_deployment(country_code)
        if deployment is None:
            print(f'Deployment for {country_code} not found, trying with global model')
            deployment = get_deployment('all')

        if deployment is None:
            print(f'No deployment found for {country_code}')
            continue

        print('FOUND EXISTING DEPLOYMENT: {}: {}'.format(deployment.id, deployment))
        print('DEPLOYMENT MODEL: {!r}'.format(deployment.model))
        model: dr.Model = dr.Model.get(deployment.model['project_id'], deployment.model['id'])
        print('FOUND EXISTING MODEL: {}: {}'.format(model.id, model))
        threshold = model.get_roc_curve(dr.enums.CHART_DATA_SOURCE.CROSSVALIDATION).get_best_f1_threshold()

        prediction_outfile = f'data/{country_code}_prediction_water_points.csv'
        rows = DF.Flow(
            DF.load(BASE_PATH + 'prediction-predict/datapackage.json'),
            DF.filter_rows(lambda r: country_code in ('all', r['clean_country_id'])),
            DF.add_field('fold', 'date', None),
            DF.update_resource(-1, path=prediction_outfile),
            DF.dump_to_path(outdir / 'prediction'),
            DF.select_fields(['wpdx_id', 'status_id']),
        ).results()[0][0]

        for years in YEARS_RANGE:
            pred_file = f'data/{country_code}_water_points_{years}y.csv'
            pred_dir = outdir / f'{years}y'
            DF.Flow(
                DF.load(str(outdir / 'prediction' / 'datapackage.json')),
                update_dates(years),
                DF.update_resource(-1, path=pred_file),
                DF.dump_to_path(pred_dir),
            ).process()

            predictions = deployment.predict_batch(
                str(pred_dir / pred_file),
                download_timeout=TIMEOUT, download_read_timeout=TIMEOUT, upload_read_timeout=TIMEOUT
            )
            predictions = predictions.to_dict('records')
            assert len(predictions) == len(rows), f'Expected {len(rows)} predictions, got {len(predictions)}'
            predictions = [
                {
                    'wpdx_id': row['wpdx_id'],
                    'status_id': row['status_id'],
                    f'prediction_yes_{years}y': pred['status_id_Yes_PREDICTION'],
                    f'prediction_no_{years}y': pred['status_id_No_PREDICTION'],
                    f'predicted_status_{years}y': 'Yes' if pred['status_id_Yes_PREDICTION'] >= threshold else 'No',
                } for row, pred in zip(rows, predictions)
            ]

            for p in predictions:
                prediction_map.setdefault(p['wpdx_id'], dict()).update(p)
        
        for point in prediction_map.values():
            values = (
                point['status_id'] == 'Yes',
                point['predicted_status_0y'] == 'Yes',
                point['predicted_status_2y'] == 'Yes'
            )
            point['predicted_category'] = CATEGORIES[values]

    print('PREDICTIONS SAVED')
    DF.Flow(
        prediction_map.values(),
        DF.dump_to_path(OUTPUT_PATH)
    ).process()
    return OUTPUT_PATH + '/datapackage.json'
