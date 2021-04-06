
import os
from file_utils import get_file


modac_collection_path='https://modac.cancer.gov/api/v2/dataObject/NCI_DOE_Archive/JDACS4C/JDACS4C_Pilot_3/pathology-reports-hierarchical-self-attention-network-hisan'
model = 'HiSAN_model.tar.gz'

model_url = os.path.join( modac_collection_path, model)

print(model_url)

get_file('hisan-trained-model.tar.gz', model_url, datadir = 'hisan-trained-model', untar = True)
