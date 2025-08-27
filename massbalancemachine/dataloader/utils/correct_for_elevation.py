"""
This method corrects the temperature and precipitation data for elevation differences and correction factors.
This factors can be glacier specific, when given as a dictionary or as a constant value for all glaciers.

Date Created: 16/12/2024
"""

import pandas as pd

def correct_T_P(df: pd.DataFrame,
                temp_grad: float = -6.5 / 1000,
                dpdz: float = 1.5 / 10000,
                gl_specific: bool = False,
                c_prec_dic: dict = {},
                t_off_dic: dict = {},
                c_prec: float = 1.434,
                t_off: float = 0.617) -> pd.DataFrame:
    """Applies temperature and precipitation corrections to the DataFrame.

    Args:
        df (pd.DataFrame): Input data frame in monthly format.
        temp_grad (float, optional): temperature gradient. Defaults to -6.5/1000 [deg/1000m].
        dpdz (float, optional): Precipitation increase in % per 100m. Defaults to 1.5/10000.
        gl_specific (bool, optional): Boolean to indicate if glacier-specific correction factors are used. Defaults to False.
        c_prec_dic (dict, optional): Dictionary with glacier-specific precipitation correction factors. Defaults to {}.
        t_off_dic (dict, optional): Dictionary with glacier-specific temperature offset factors. Defaults to {}.
        c_prec (float, optional): Constant precipitation correction factor. Defaults to 1.434.
        t_off (float, optional): Constant temperature offset. Defaults to 0.617.
        
    Returns:
        pd.DataFrame: data frame with new corrected temperature and precipitation columns.
    """
    # Apply temperature gradient correction
    df['t2m_corr'] = df['t2m'] + (df['ELEVATION_DIFFERENCE'] * temp_grad)

    if gl_specific:
        # Vectorized application for GLACIER-specific correction factors
        glacier_specific_prec = df['GLACIER'].map(c_prec_dic).fillna(1)
        df['tp_corr'] = df['tp'] * glacier_specific_prec

        # Vectorized application for temperature bias offset
        glacier_specific_toff = df['GLACIER'].map(t_off_dic).fillna(1)
        df['t2m_corr'] += glacier_specific_toff
    else:
        df['tp_corr'] = df['tp'] * c_prec
        df['t2m_corr'] += t_off

    # Apply elevation correction factor
    df['tp_corr'] += df['tp_corr'] * (df['ELEVATION_DIFFERENCE'] * dpdz)

    return df
