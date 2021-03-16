
import os
from file_utils import get_file


modac_collection_path='https://modac.cancer.gov/api/v2/dataObject/NCI_DOE_Archive/JDACS4C/JDACS4C_Pilot_3/multitask_cnn/'
model = 'mt_cnn_model.h5'

model_url = os.path.join( modac_collection_path, model)

print(model_url)

get_file('mt_cnn_model.h5', model_url, datadir = '.')
