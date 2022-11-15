
import pandas
ce = pandas.read_csv("./resource/v2/csv/embedding.csv")
da = pandas.read_csv("./resource/v2/csv/test.csv")

ce_list = []
for _, item in da.iterrows():

    if(item['vote']=='lower'): ce_list += [ce.iloc[0,:].values]
    if(item['vote']=='higher'): ce_list += [ce.iloc[1,:].values]
    continue

ce_df = pandas.DataFrame(ce_list)
ce_df.columns = ce.columns
da = pandas.concat([da, ce_df],axis=1)
da.to_csv("test_ce.csv", index=False)
