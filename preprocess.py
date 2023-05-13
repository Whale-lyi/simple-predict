def replace_data(df, unused_col):
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and col not in unused_col:
            quantile_2 = df[col].quantile(0.02)
            quantile_98 = df[col].quantile(0.98)
            df.loc[df[col] < quantile_2, col] = quantile_2
            df.loc[df[col] > quantile_98, col] = quantile_98
    return df


