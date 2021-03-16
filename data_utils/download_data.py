
import os
from file_utils import get_file


modac_collection_path='https://modac.cancer.gov/api/v2/dataObject/NCI_DOE_Archive/JDACS4C/JDACS4C_Pilot_3/ml_ready_pathology_reports/'
metadata = 'raw_text_histo_metadata.csv'
reports = 'raw_text_pathology_reports.tar.gz'

metadata_url = os.path.join( modac_collection_path, metadata)
reports_url = os.path.join( modac_collection_path, reports)

print(metadata_url)
print(reports_url)

get_file('histo_metadata.csv', metadata_url, datadir = 'modac_data')
get_file('features_full.tar.gz', reports_url, datadir = 'modac_data', untar = True)


