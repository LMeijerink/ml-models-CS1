import numpy as np
import pandas as pd


def prepare_data(df):
    dys_base_var = 'dysfagie_umcgshortv2_bsl'
    dys_outc_var = 'dysfagie_umcgshortv2_m06'
    xer_base_var = 'q41_bsl'
    xer_outc_var = 'q41_m06'

    # tumor location
    mask = df['loctum_cat'].notna()
    df.loc[mask, 'primary_tumor_location_oral_cavity'] = df.loc[mask, 'loctum_cat'] == "Oral cavity"
    df.loc[mask, 'primary_tumor_location_larynx'] = df.loc[mask, 'loctum_cat'] == "Larynx"
    df.loc[mask, 'primary_tumor_location_pharynx'] = df.loc[mask, 'loctum_cat'].isin(
        ['Oropharynx', 'Nasopharynx', 'Hypopharynx', 'Neopharynx'])

    # dysphagia baseline
    mask = df[dys_base_var].notna()
    df.loc[mask, 'dysphagia_at_baseline_0_1'] = df.loc[mask, dys_base_var] == 'Grade 0-1 (regular diet)'
    df.loc[mask, 'dysphagia_at_baseline_2'] = df.loc[mask, dys_base_var] == 'Grade 2 (soft food)'
    df.loc[mask, 'dysphagia_at_baseline_3_5'] = df.loc[
                                                    mask, dys_base_var] == 'Grade 3-5 (liquids only or tube feeding dependent)'

    # dysphagia 6 months
    mask = df[dys_outc_var].notna()
    df.loc[mask, 'dysphagia_at_m6_2plus'] = df.loc[mask, dys_outc_var].isin(['Grade 2 (soft food)',
                                                                             'Grade 3-5 (liquids only or tube feeding dependent)'])
    df.loc[mask, 'dysphagia_at_m6_3plus'] = df.loc[
                                                mask, dys_outc_var] == 'Grade 3-5 (liquids only or tube feeding dependent)'

    # xerostomia baseline
    mask = df[xer_base_var].notna()
    df.loc[mask, 'xerostomia_at_baseline_1'] = df.loc[mask, xer_base_var] == 'Helemaal niet'
    df.loc[mask, 'xerostomia_at_baseline_2'] = df.loc[mask, xer_base_var] == 'Een beetje'
    df.loc[mask, 'xerostomia_at_baseline_3_4'] = df.loc[mask, xer_base_var].isin(['Nogal', 'Heel erg'])

    # xerostomia 6 months
    mask = df[xer_outc_var].notna()
    df.loc[mask, 'xerostomia_at_m6_2plus'] = df.loc[mask, xer_outc_var].isin(
        ['Nogal', 'Heel erg'])  # CHECK!! 2 plus komt niet overeen met baseline cat
    df.loc[mask, 'xerostomia_at_m6_3plus'] = df.loc[mask, xer_outc_var] == "Heel erg"

    # dose parameters
    df = df.assign(
        rt_dmean_parotid_low=df['parotid_low_dmean'],
        rt_dmean_parotid_high=df['parotid_high_dmean'],
        rt_dmean_both_submandibulars=df['submandibulars_dmean'],
        rt_dmean_oral_cavity=df['oralcavity_ext_dmean'],
        rt_dmean_pcm_superior=df['pcm_sup_dmean'],
        rt_dmean_pcm_medius=df['pcm_med_dmean'],
        rt_dmean_pcm_inferior=df['pcm_inf_dmean']
    )

    # transformed features
    df = df.assign(
        rt_sqrt_dmean_ipsilateral_parotid_sqrt_dmean_contralateral_parotid=np.sqrt(df['rt_dmean_parotid_low']) +
                                                                           np.sqrt(df['rt_dmean_parotid_high'])
    )

    return (df)