
import pandas as pd
from pysnptools.snpreader import Bed

from config_manager import get_directory

def snp_reader(bedfile=None):
    snps = Bed(get_directory.adni_bedfile, count_A1=False)

    snp_data = snps.read()
    sids = snp_data.sid
    iids = snp_data.iid
    ids = []
    for i,iid in enumerate(iids):
        ids.append(iids[i][1])
    snp_df = pd.DataFrame(data=snp_data.val, index=ids,columns=sids, dtype='int8')
    return snp_df