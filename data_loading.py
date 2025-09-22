import pandas as pd
import numpy as np

def _load_and_clean_single_file(filepath):
    """Loads a single CSV file, performs cleaning, and returns a DataFrame."""
    try:
        df = pd.read_csv(filepath, dtype=str) 
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Error: Could not find '{filepath}'. Please ensure it's in the same directory."
        ) from e

    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].str.strip()
            
    return df

def get_f0(fwd_file, name, delivery_period):
    """Loads forward data and retrieves F0 for a specific contract."""
    dati_fwd = _load_and_clean_single_file(fwd_file)
    
    fwd_filtered = dati_fwd[
        (dati_fwd['Name'] == name) &
        (dati_fwd['DeliveryPeriod'] == delivery_period)
    ]
    
    if fwd_filtered.empty:
        raise ValueError(f"Could not find forward contract with Name='{name}' and DeliveryPeriod='{delivery_period}'.")
        
    return pd.to_numeric(fwd_filtered['SettlementPrice']).iloc[0]

def get_option_data(opt_file, underlying, option_type='C', ref_date_str='2019-11-19'):
    """Loads and filters option data for a specific underlying."""
    dati_opt = _load_and_clean_single_file(opt_file)

    opt_filtered = dati_opt[
        (dati_opt['Underlying'] == underlying) &
        (dati_opt['Type'] == option_type)
    ].copy()

    if opt_filtered.empty:
        raise ValueError(f"Could not find '{option_type}' options for underlying '{underlying}'.")

    ref_date = pd.to_datetime(ref_date_str)
    opt_filtered['MaturityDate'] = pd.to_datetime(opt_filtered['ExpiryDate'])
    opt_filtered['T'] = (opt_filtered['MaturityDate'] - ref_date).dt.days / 365.0
    
    T = opt_filtered['T'].values
    K = pd.to_numeric(opt_filtered['Strike']).values
    P = pd.to_numeric(opt_filtered['SettlementPrice']).values
    
    return T, K, P

def get_historical_log_returns(
    hist_file: str, 
    date_col: str, 
    price_col: str, 
    cutoff_date_str: str
    ) -> pd.Series:
    """
    Loads historical prices, calculates log-returns, and filters by date.
    """
    df_hist = _load_and_clean_single_file(hist_file)
    
    df_hist[date_col] = pd.to_datetime(df_hist[date_col])
    df_hist[price_col] = pd.to_numeric(df_hist[price_col])
    
    cutoff_date = pd.to_datetime(cutoff_date_str)
    df_hist = df_hist[df_hist[date_col] <= cutoff_date].set_index(date_col)
    
    log_returns = np.log(df_hist[price_col] / df_hist[price_col].shift(1))
    
    return log_returns.dropna()
