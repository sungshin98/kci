import pandas as pd

df = pd.read_csv('./IBI/Session01/Sess01_script01_User001F.csv')
df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time', inplace = True)
df = df.resample('0.25S').interpolate()

df.reset_index(inplace=True)

print(df_resampled)