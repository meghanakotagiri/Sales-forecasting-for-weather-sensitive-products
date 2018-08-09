import pandas as pd
import numpy as np

#separating store-item combinations which have zero sales and which have non-zero sales
def create_vaild_item_store_combinations(_df):
    df = _df.copy()
    df['log1p'] = np.log(df['units'] + 1)
    g = df.groupby(["store_nbr", "item_nbr"])['log1p'].mean()
    g2=g.copy()
    g = g[g > 0.0]
    g2 = g2[ g2 == 0]
    store_nbrs = g.index.get_level_values(0)
    item_nbrs = g.index.get_level_values(1)
    
    store_item_nbrs = sorted(zip(store_nbrs, item_nbrs), key = lambda t: t[1] * 10000 + t[0] )

    inv_store_nbrs = g2.index.get_level_values(0)
    inv_item_nbrs = g2.index.get_level_values(1)
    
    inv_store_item_nbrs = sorted(zip(inv_store_nbrs, inv_item_nbrs), key = lambda t: t[1] * 10000 + t[0] )


    with open(store_item_nbrs_path, 'w') as f: 
        f.write("store_nbr,item_nbr\n")
        for sno, ino in store_item_nbrs:
            f.write("{},{}\n".format(sno, ino))

    with open(invalid_store_item_nbrs_path, 'w') as f: 
        f.write("store_nbr,item_nbr\n")
        for sno, ino in inv_store_item_nbrs:
            f.write("{},{}\n".format(sno, ino))


store_item_nbrs_path = 'store_item_nbrs.csv'
invalid_store_item_nbrs_path = 'invalid_store_item_nbrs.csv'
df_train = pd.read_csv("data/train.csv")
create_vaild_item_store_combinations(df_train)