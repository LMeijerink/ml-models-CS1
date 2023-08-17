import numpy as np
import re
import pandas as pd
import src.utils as utils
import os

dys_base_var = 'dysfagie_umcgshortv2_bsl'
dys_outc_var = 'dysfagie_umcgshortv2_m06'
xer_base_var = 'q41_bsl'
xer_outc_var = 'q41_m06'


def remap_dysf_3cat(df):
    # remapping dysfagia to 3 categories, only necessary for dev data
    target_dict = {
        "Grade 0-1 (regular diet)": "Grade 0-1 (regular diet)",
        "Grade 2 (soft food)": "Grade 2 (soft food)",
        "Grade 3 (liquids only )": "Grade 3-5 (liquids only or tube feeding dependent)",
        "Grade 4-5 (tube feeding dependent)": "Grade 3-5 (liquids only or tube feeding dependent)"}

    df[dys_outc_var] = df['dysfagie_umcgshort_m06'].replace(target_dict)  # 6 months
    df[dys_base_var] = df['dysfagie_umcgshort_bsl'].replace(target_dict)  # baseline
    return df


def prepare_baseline(df):
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
    df.loc[mask, 'xerostomia_at_m6_2plus'] = df.loc[mask, xer_outc_var].isin(['Nogal', 'Heel erg'])
    df.loc[mask, 'xerostomia_at_m6_3plus'] = df.loc[mask, xer_outc_var] == "Heel erg"

    # dose parameters
    df = df.rename(columns={'parotid_low_dmean': 'rt_dmean_parotid_low',
                            'parotid_high_dmean': 'rt_dmean_parotid_high',
                            'submandibulars_dmean': 'rt_dmean_both_submandibulars',
                            'oralcavity_ext_dmean': 'rt_dmean_oral_cavity',
                            'pcm_sup_dmean': 'rt_dmean_pcm_superior',
                            'pcm_med_dmean': 'rt_dmean_pcm_medius',
                            'pcm_inf_dmean': 'rt_dmean_pcm_inferior'})

    # transformed features
    df = df.assign(
        rt_sqrt_dmean_ipsilateral_parotid_sqrt_dmean_contralateral_parotid=np.sqrt(df['rt_dmean_parotid_low']) +
                                                                           np.sqrt(df['rt_dmean_parotid_high'])
    )

    week_month_pattern = r".*_(w|m)\d{2}$"
    drop_cols = ['research.id']
    col_select = [c for c in df.columns.values if not (re.match(week_month_pattern, c) and not c in drop_cols)]
    df = df[col_select]
    return df


if __name__ == "__main__":
    dev_df = pd.read_spss(os.path.join(utils.DATA_DIR_RAW, "CITOR.development.data.sav"))
    val_df = pd.read_spss(os.path.join(utils.DATA_DIR_RAW, "CITOR.validatie.data.sav"))

    # everything lower case so we don't have to worry about that
    dev_df.columns = dev_df.columns.str.lower()
    val_df.columns = val_df.columns.str.lower()

    # dysfagie is remapped to 3 categories for the development data
    dev_df = remap_dysf_3cat(dev_df)

    # prepare baseline features and targets, drop features that are in the future so should not be used as predictors
    dev_df = prepare_baseline(dev_df)
    val_df = prepare_baseline(val_df)

    dev_df.to_pickle(os.path.join(utils.DATA_DIR_INTERIM, 'dev_df.pkl'))
    val_df.to_pickle(os.path.join(utils.DATA_DIR_INTERIM, 'val_df.pkl'))
