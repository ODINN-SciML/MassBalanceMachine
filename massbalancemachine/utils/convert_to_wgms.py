def convert_to_wgms(df, column_names_dates, column_names_SMB):
    """
    In case the dataset has one record, that belongs to a stake, for multiple measurements in a single hydrological
    year, i.e., winter, summer, annual, this function converts this single record to an individual record for each of
    the measurements in that specific hydrological year. For each record, also an identifier is added. The identifier
    can be used to aggregate the different measurement, either monthly or seasonally.

    Assumed is that a record has the following sort of structure:
        Stake ID, ..., Measurement Date 1, Measurement Date 2, Measurement Date 3, ..., Winter SMB, Summer SMB, Annual SMB, ...

    After melting the record, the structure will be as follows:
        Stake ID, ..., Measurement Date 1, Winter SMB, ..., BW
        Stake ID, ..., Measurement Date 2, Summer SMB, ..., BS
        Stake ID, ..., Measurement Date 3, Annual SMB, ..., BA

    Args:
        df (pandas dataframe): Dataframe containing records that have multiple measurements for a single stake
    Returns:
        df (pandas dataframe): Dataframe with the raw data is melted to a wgms-like format so that a single stake can have
        multiple records that correspond to different measurement periods.
    """



    return None