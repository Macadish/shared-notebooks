import pandas as pd
cd ~/shared-notebooks/notebooks/HDB_analysis
pc2string = lambda x: '{0:06d}'.format(x)
data = pd.read_csv('./data/raw/sggeocode.csv').rename(columns={"Unnamed: 0":"postalcode"})
data['postalcode'] = data['postalcode'].apply(pc2string)
data = data.set_index('postalcode')
data['sectorcode'] = data.index.map(lambda x:x[:2])
data.to_csv('./data/processed/sggeocode_withsectorcode.csv', sep='\t')
