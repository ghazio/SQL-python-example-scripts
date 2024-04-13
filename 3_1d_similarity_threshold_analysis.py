
from ast import Delete
from datetime import datetime
import os
import numpy as np
import pandas as pd
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import yaml
from pathlib import Path
from sqlalchemy import create_engine, inspect
from sqlalchemy.types import NUMERIC, TEXT
from KDEpy import FFTKDE
from argparse import ArgumentParser

#import sshtunnel
sys.path.insert(0, "../scripts")
from utils import *

def n_dups_cal(x):
    #print(thres)
    return (x>=thres).sum()


def figname_generator_thresh(figurename,n_params):
    return f"{figurename}_{thres}.png"

def figname_generator_(figurename,n_params):
    return f"{figurename}_{thres}_{n_params}.png"

#print(figname_generator_("", 2))

"""
Inputs: data about similarity scores, entity(Permit, or Lab), and group of combinations
Output: returns the number of duplications per Entity for the given group
"""
def get_dupe_rate(df, entity, group,thres):
    dup_rate = df[df['group'] == group].groupby(entity)['similarity'].agg([('n_samples',"count"),('n_dups',n_dups_cal)]).reset_index()   
    dup_rate['dup_rate'] = (
        1000*dup_rate['n_dups'] / dup_rate['n_samples']).replace(np.inf, 0)
    dup_rate['group'] = group
    dup_rate['thres'] = thres

    return dup_rate

def parse_config(config):
    
    # read in configured parameters
    with open(Path().resolve().parent / config, 'r') as file:
    #with open(Path().resolve().parent / 'config.yml', 'r') as file:
        configs = yaml.safe_load(file)

   # read in configured parameters
    with open(Path().resolve().parent / config, 'r') as file:
    #with open(Path().resolve().parent / 'config.yml', 'r') as file:
        configs = yaml.safe_load(file)

    global STATES        
    STATES = configs['states']


    # name of the table on the Postgres that stores the DMR values matrix
    global DMR_MATRIX_NAME
    DMR_MATRIX_NAME = configs['dmr_table_name']

    # these combinations tables must exist on the Postgres server
    G1_COMBS = 'group1_'+configs['group_combs_name']
    G2_COMBS = 'group2_'+configs['group_combs_name']
    G3_COMBS = 'group3_'+configs['group_combs_name']
    G4_COMBS = 'group4_'+configs['group_combs_name']
    G5_COMBS = 'group5_'+configs['group_combs_name']
    global combs_dict
    combs_dict = {'group1':G1_COMBS,'group2':G2_COMBS,'group3':G3_COMBS,'group4':G4_COMBS,'group5':G5_COMBS}

    # these results tables must exist on the Postgres server
    G1_RESULTS = 'group1_'+configs['group_results_name']
    G2_RESULTS = 'group2_'+configs['group_results_name']
    G3_RESULTS = 'group3_'+configs['group_results_name']
    G4_RESULTS = 'group4_'+configs['group_results_name']
    G5_RESULTS = 'group5_'+configs['group_results_name']
    global results_dict 
    results_dict = {'group1':G1_RESULTS,'group2':G2_RESULTS,'group3':G3_RESULTS,'group4':G4_RESULTS,'group5':G5_RESULTS}

    
    global thres 
    thres = configs['similarity_threshold']
    
    global num_unique
    num_unique = configs['num_unique']
    
    global gini
    gini = configs['gini']
    
    global n_shared_params 
    n_shared_params = configs['min_shared_params']
    
    global pair_sample_rate 
    pair_sample_rate = configs['pair_sample_rate']

    global ROOT 
    ROOT = os.path.expanduser(Path(configs['project_directory']))  # expands '~' in path if necessary
    #NEED TO RESOLVE THIS SO THAT RESULTS DIRECTORY IS ARRANGED BY:
        # GINI-NUM_UNIQUE-THRESHOLD
    global OUT
    thres = 0.9
    OUT = os.path.join(configs['output_directory'],f'results_gini_{gini}_unique_{num_unique}_similarity_{thres}_pair_samplerate{pair_sample_rate}')
   
    global prop_shared_params
    prop_shared_params = configs['prop_shared_params']
    


if  __name__ == '__main__':
    start = datetime.now()
    
    parser = ArgumentParser()

    parser.add_argument('config_file', default=None)
    args = parser.parse_args()
    parse_config(args.config_file)
    print(results_dict)
    print(combs_dict)
    print(DMR_MATRIX_NAME)


    print("Loading data")
    GROUPS = combs_dict.keys()
  
    #make a postgres+pyscopg2 engine
    engine_str ='postgresql+psycopg2://'+os.environ['postgres_user']+':'+os.environ['postgres_pass']+'@localhost:5432/dmr-integrity'
    engine = create_engine(engine_str)
    pg_conn = engine.connect()
    dbapi_conn = pg_conn.connection   
    pair_dfs = []
    dupe_rates_df = pd.DataFrame()
    #STATES = ["AK","KY"]
    for state in tqdm.tqdm(STATES):
        print(f"STARTING {state}")
        #DO FIRST GROUP
        sql = "SELECT * FROM {0} WHERE permit_state = '{1}' and n_shared_params >= {2} and prop_params_shared >= {3}".format(results_dict['group1'],state, n_shared_params, prop_shared_params)
        df = pd.read_sql(sql, con=dbapi_conn)
        df['state'] = state
        df['group'] = 'Same permit'
        pair_dfs.append(df[['similarity','group','state', 'n_shared_params', 'max_n_value_params']].sample(frac=pair_sample_rate))
        for thres in [0.8,0.85,0.9,0.95,1]:
            dup_rate = get_dupe_rate(df, 'npdes_permit_id', 'Same permit',thres)
            dupe_rates_df = dupe_rates_df.append(dup_rate, ignore_index=True)
        
        #DO SECOND GROUP
        sql = "SELECT * FROM {0} WHERE permit_state = '{1}' and n_shared_params >= {2} and prop_params_shared >= {3}".format(results_dict['group2'],state, n_shared_params, prop_shared_params)
        df = pd.read_sql(sql, con=dbapi_conn)
        df['group'] = 'Same lab same permit'
        df['state'] = state
        pair_dfs.append(df[['similarity','group','state', 'n_shared_params', 'max_n_value_params']].sample(frac=pair_sample_rate))

        for thres in [0.8,0.85,0.9,0.95,1]:
            dup_rate = get_dupe_rate(df, ['lab_name', 'npdes_permit_id'], 'Same lab same permit',thres)
            dupe_rates_df = dupe_rates_df.append(dup_rate, ignore_index=True)
        
        #DO THIRD GROUP
        sql = "SELECT * FROM {0} WHERE permit_state = '{1}' and n_shared_params >= {2} and prop_params_shared >= {3}".format(results_dict['group3'],state, n_shared_params, prop_shared_params)
        df = pd.read_sql(sql, con=dbapi_conn)
        df['state'] = state
        df['group'] = 'Same lab diff permit'
        pair_dfs.append(df[['similarity','group','state', 'n_shared_params', 'max_n_value_params']].sample(frac=pair_sample_rate))
        for thres in [0.8,0.85,0.9,0.95,1]:
            dup_rate = get_dupe_rate(df, ['lab_name', 'npdes_permit_id'], 'Same lab diff permit',thres)
            dupe_rates_df = dupe_rates_df.append(dup_rate, ignore_index=True)


        #CALCULATE STRATIFIED PLACEBO 
        sql = "SELECT * FROM {0} WHERE permit_state = '{1}' and n_shared_params >= {2} and prop_params_shared >= {3}".format(results_dict['group5'],state, n_shared_params, prop_shared_params)
        df = df = pd.read_sql(sql, con=dbapi_conn)
        df['state'] = state        
        df['group'] = 'Placebo stratified'
        pair_dfs.append(df[['similarity','group','state', 'n_shared_params', 'max_n_value_params']].sample(frac=pair_sample_rate))

        for thres in [0.8,0.85,0.9,0.95,1]:
            dup_rate = get_dupe_rate(df, ['lab_name', 'npdes_permit_id'], 'Placebo stratified',thres)
            dupe_rates_df = dupe_rates_df.append(dup_rate, ignore_index=True)
        del df

    result_df = pd.concat(pair_dfs, ignore_index=True)
    #pdb.set_trace()

    ### FIND OUT HOW THE Number of Duplications are CHANGING ACROSS GROUP
    unique_vals = dupe_rates_df.drop_duplicates("group").reset_index()
    unique_vals = unique_vals[["group","n_samples"]]



    unique_vals["0.8_n_dups"] = 0
    unique_vals["0.8_dup_rate"] = 0.0
    unique_vals["0.8_unique_permits"] = 0.0
    unique_vals["0.8_unique_labs"] = np.nan


    unique_vals["0.85_n_dups"] = 0
    unique_vals["0.85_dup_rate"] = 0.0
    unique_vals["0.85_unique_permits"] = 0.0
    unique_vals["0.85_unique_labs"] = np.nan


    unique_vals["0.9_n_dups"] = 0
    unique_vals["0.9_dup_rate"] = 0.0
    unique_vals["0.9_unique_permits"] = 0.0
    unique_vals["0.9_unique_labs"] = np.nan


    unique_vals["0.95_n_dups"] = 0
    unique_vals["0.95_dup_rate"] = 0.0
    unique_vals["0.95_unique_permits"] = 0.0
    unique_vals["0.95_unique_labs"] = np.nan


    unique_vals["1_n_dups"] = 0
    unique_vals["1_dup_rate"] = 0.0
    unique_vals["1_unique_permits"] = 0.0
    unique_vals["1_unique_labs"] = np.nan

    unique_labs_g1 = {}
    unique_labs_g2 = {}

    #pdb.set_trace()
    
    for i,row in unique_vals.iterrows():
        unique_vals.at[i,"n_samples"] = dupe_rates_df[dupe_rates_df['group']==row['group']][dupe_rates_df['thres']==thres]['n_samples'].sum()
        for thres in [0.8,0.85,0.9,0.95,1]:
            dfx = dupe_rates_df[(dupe_rates_df['group']==row['group']) & (dupe_rates_df['thres']==thres)]
            ### THIS SETS UNIQUE LABS
            if i == 1: 
                unique_vals.at[i,f"{thres}_unique_labs"] =  dfx[dfx['n_dups']>0].drop_duplicates('lab_name').shape[0]
                unique_labs_g1[thres]=(dfx[dfx['n_dups']>0])

            if i == 2: 
                unique_vals.at[i,f"{thres}_unique_labs"] =  dfx[dfx['n_dups']>0].drop_duplicates('lab_name').shape[0]
                unique_labs_g2[thres]=(dfx[dfx['n_dups']>0])
            ### THIS SETS UNIQUE PERMITS FOR DUPLICATES
            unique_vals.at[i,f"{thres}_unique_permits"] = dfx[dfx['n_dups']>0].drop_duplicates('npdes_permit_id').shape[0]
            assert dupe_rates_df[dupe_rates_df['group']==row['group']][dupe_rates_df['thres']==thres]["n_dups"].sum() == dfx["n_dups"].sum()
            unique_vals.at[i,f"{thres}_n_dups"] = dupe_rates_df[dupe_rates_df['group']==row['group']][dupe_rates_df['thres']==thres]["n_dups"].sum()
            unique_vals.at[i,f"{thres}_dup_rate"] = dupe_rates_df[dupe_rates_df['group']==row['group']]\
                                                [dupe_rates_df['thres']==thres]["n_dups"].sum()\
                                                      / dupe_rates_df[dupe_rates_df['group']==row['group']][dupe_rates_df['thres']==thres]['n_samples'].sum()*1000
    unique_vals.to_csv("duplication_rate_analysis.csv")
    print(unique_vals)
    #pdb.set_trace()
    g1_labs = {}
    g2_labs = {}
    big_setg1 = set()
    big_setg2 = set()

    for thres in reversed([0.8,0.85,0.9,0.95,1]):
        #get name of the list
        placeholder1 = unique_labs_g1[thres].drop_duplicates('lab_name')['lab_name'].tolist()
        for elem in placeholder1[:]:
            if elem in big_setg1:
                placeholder1.remove(elem)
            else:
                big_setg1.add(elem)
        g1_labs[thres] = placeholder1 + [len(placeholder1)]    

        placeholder = unique_labs_g2[thres].drop_duplicates('lab_name')['lab_name'].tolist()
        for elem in placeholder[:]:
            if elem in big_setg2:
                placeholder.remove(elem)
            else:
                big_setg2.add(elem)
        g2_labs[thres] = placeholder + [len(big_setg2)] + [f"Shared labs: {len(big_setg2.intersection(big_setg1))}"]
        print(len(g1_labs[thres])," vs ",len(g2_labs[thres]),f"for thres {thres}", f"intersection size {len(big_setg2.intersection(big_setg1))}")
    
    g1_labs = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in g1_labs.items() ]))
    g2_labs = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in g2_labs.items() ]))
    g1_labs.to_csv("group2_labs.csv")
    g2_labs.to_csv("group3_labs.csv")
    #pdb.set_trace()
    labs_only = dupe_rates_df[["lab_name","npdes_permit_id"]]

    
    print('Done')
    print("Time Elapsed: ", datetime.now()-start)
