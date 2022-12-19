import streamlit as st
import pandas as pd
import numpy as np
from itertools import islice
pd.options.mode.chained_assignment = None

st.title('VUMC Diagnoses')
st.markdown('## Welcome to the VUMC Diagnoses Dashboard')
st.markdown('- In order to get started using this quality control dashboard please upload the full Redcap data export as a .csv file in the space below.')
data_file = st.file_uploader('Choose Data File')
RC_crosswalk_url = "https://github.com/smavt93/Diagnoses/blob/main/Patient%20info%20(RC).xlsx?raw=true"
vumc_item_syn_url = "https://github.com/smavt93/Diagnoses/blob/main/VUMC%20RC%20Crosswalk.xlsx?raw=true"

if data_file is not None:
    data_load_state = st.text('Loading data...')
    data_load_state.text('Data Loading is done!')
    
################################################################################## Opening Files ############################################################################################
if data_file is not None:
    full_db = pd.read_csv(data_file, index_col=['subject_id'])
    filtered_full_db_1 = full_db[(full_db['scid_iscomplete'] == 'Yes') & (full_db['dx1'] != 47)]
    filtered_full_db = filtered_full_db_1.iloc[:770, :]
    vumc_item_syn_db = pd.read_excel(vumc_item_syn_url)
    crosswalk_db = pd.read_excel(RC_crosswalk_url)

################################################################################## Creating Lists ##############################################################################################

    # Getting the RC diag variable names to create the dictionary crosswalk
    rc_diag_names = crosswalk_db.rc_variables.tolist() #e.g., dx1, dx2, etc.
    rc_diag_names_cleaned = [x for x in rc_diag_names if str(x) != 'nan']

    # Getting the # values and their corresponding syndrome names
    rc_diag_values = crosswalk_db.dx_values.tolist() # e.g., 1, 2, 3
    syndrome_names = crosswalk_db.syndrome_names.tolist() # e.g., Major Depressive Disorder, etc.

    # Creating a dictionary with the previous two lists ({1: "Major Depressive Disorder", etc.})
    rc_diag_crosswalk_dict = {rc_diag_values[i]: syndrome_names[i] for i in range(len(rc_diag_values))}

    # List of syndromes that are not represented in the VUMC syndrome list
    exclusion_list = crosswalk_db.syndromes_w_na.tolist()
    cleaned_exclusion_list = [x for x in exclusion_list if str(x) != 'nan']

    # List of syndromes that have only one comparitor
    easy_syndrome_list = crosswalk_db.syndromes_w_one_comparison.tolist() # E.g., Major Depressive Disorder, etc.
    cleaned_easy_syndrome_list = [x for x in easy_syndrome_list if str(x) != 'nan']

    # List of syndromes that need two comparitors
    complex_syndrome_list = crosswalk_db.syndromes_w_two_comparisons.tolist() # e.g., Persistent Depressive Disorder
    cleaned_complex_syndrome_list = [x for x in complex_syndrome_list if str(x) != 'nan']

    # Getting the next set of lists to form the last dictionary
    rc_item_diag = vumc_item_syn_db.rc_variables.tolist() # e.g., scid_a25, scid_a51, etc.
    rc_item_diag_names = vumc_item_syn_db.syndrome_names.tolist() # e.g., Current Major Depression Episode, etc.

    # Creating a dictionary with the previous two lists ({'scid_a25' : 'Current Major Depresssion Episode', etc.})
    vumc_item_crosswalk_dict = {rc_item_diag[i]: rc_item_diag_names[i] for i in range(len(rc_item_diag))}

    # List of VUMC syndrome names that have only one comparitor # This is just to select the correct column names
    non_na_vumc_syndromes = vumc_item_syn_db.syndrome_non_na.tolist() # E.g., MDD Diagnosis, etc.
    cleaned_non_na_vumc_syndromes = [x for x in non_na_vumc_syndromes if str(x) != 'nan']

    # List of VUMC syndrome names that have two comparitors # This is just to select the correct column names # This doubles as a list that is used to create lifetime diagnoses columns
    vumc_complex_syndrome_list = vumc_item_syn_db.syndrome_two.tolist() # e.g., CPDD Diagnosis, PPDD Diagnosis, etc.
    cleaned_vumc_complex_syndrome_list = [x for x in vumc_complex_syndrome_list if str(x) != 'nan']

    # Combining the dx1, dx2, ... list with the scid_a25, scid_a51, ... list to create the filter criteria for the new db
    column_filter = rc_diag_names_cleaned + rc_item_diag

    ################################################################################ Creating Specific DBs ########################################################################################

    ######## Creation of the first db ####################

    # The creation of the db with only the desired columns
    first_step_diagnoses_db = filtered_full_db.loc[:, column_filter]

    # Renaming the non dx1, ... columns with their appropriate syndrome names
    correctly_named_diagnosis_db = first_step_diagnoses_db.rename(columns=vumc_item_crosswalk_dict)

    # Replacing the "Sedative" designation in the dx columns in order to make processing easier. 
    intermediate_diagnosis_db = correctly_named_diagnosis_db.replace({'dx1':{'Sedative':48}, 'dx2': {'Sedative':48}, 'dx3': {'Sedative':48}, 
    'dx4': {'Sedative':48}, 'dx5': {'Sedative':48}, 'dx6': {'Sedative':48}, 'dx7': {'Sedative':48}, 
    'dx8': {'Sedative':48}, 'dx9': {'Sedative':48}, 'dx10': {'Sedative':48}, 'dx11': {'Sedative':48}})

    # forcing the designation of the dx columns to be interpreted as floats for future parsing.
    for i in range(2,12):
        intermediate_diagnosis_db[f'dx{i}'] = pd.to_numeric(intermediate_diagnosis_db[f'dx{i}'])

    diagnoses_db = intermediate_diagnosis_db.fillna(0)

    ######### Creation of the Lifetime DB #####################

    # Now creating a lifetime DB that we will append to the final db
    full_lifetime_db = diagnoses_db.loc[:, cleaned_vumc_complex_syndrome_list]

    # Making the column names easier to work with for the eventual creation of the new lifetime columns # Suffixes are easier to remove than the variable prefix
    current_lifetime_db = full_lifetime_db.rename(columns = lambda x: x + '_c' if x.startswith("C") else x)
    total_lifetime_db = current_lifetime_db.rename(columns= lambda x: x + "_p" if x.startswith("P") else x + "_l" if x.startswith("L") else x)
    # This takes away the preceeding "P" and "C"
    slim_lifetime_db = total_lifetime_db.rename(columns = lambda x: x[1:])

    # This is a list that we will iterate through to create the lifetime columns
    lifetime_column_names_w_suffix = slim_lifetime_db.columns.tolist()

    # Selecting the current only items so that we can get the name of the lifetime column # Could have been past columns just needed to collect every other item
    current_only_names = []
    for x in islice(lifetime_column_names_w_suffix, 0, None, 2):
        current_only_names.append(x)

    # Loop created to form the lifetime columns. This is adding these columns to the end of the lifetime_db DF
    count_current = 0
    count_lifetime = 0
    Lifetime_copy = slim_lifetime_db.copy(deep=True)
    while count_current < len(current_only_names):
        while count_lifetime < len(lifetime_column_names_w_suffix):
            syndrome_name_w_stuff = current_only_names[count_current]
            if "Diagnosis_c" in syndrome_name_w_stuff:
                syndrome_name = syndrome_name_w_stuff.removesuffix("Diagnosis_c")
            else:
                syndrome_name = syndrome_name_w_stuff.removesuffix("Diagnosis_l")
            count_current += 1
            final_loop_db = Lifetime_copy
            second_index = count_lifetime + 1
            final_loop_db[str(syndrome_name) + "Lifetime Diagnosis"] = final_loop_db[[str(lifetime_column_names_w_suffix[count_lifetime]), 
            str(lifetime_column_names_w_suffix[second_index])]].max(axis = 1)
            count_lifetime += 2

    # Replacing all diagnoses of 1 as 0 just to reduce clutter. We don't really care if one person left the item empty and the other marked it 1. Essentially the same.
    clean_final_db = final_loop_db.replace(1, 0)
    # Only keeping the lifetime columns # No longer need the past or current columns
    final_db = clean_final_db.loc[:,~clean_final_db.columns.str.contains('_', case = False)]

    # Creating a list of the lifetime columns created
    lifetime_column_names_final = final_db.columns.tolist()

    ########## Creating the Joined DB (pt. 1) ####################

    # First we will want to drop the columns that have lifetime diagnoses items
    refined_diagnoses_db = diagnoses_db.drop(columns = cleaned_vumc_complex_syndrome_list, axis = 1)
    cleaned_refined_diagnosis_db = refined_diagnoses_db.iloc[:, 11:].replace(1,0)
    pt_1_join = pd.concat([refined_diagnoses_db.iloc[:, :11], cleaned_refined_diagnosis_db], axis = 1)
    joined_db_1 = pd.concat([pt_1_join, final_db], axis = 1)

    count = 0
    while count < len(syndrome_names):
        def categorise(row):
            if row['dx1'] == rc_diag_values[count]:
                return 3
            elif row['dx2'] == rc_diag_values[count]:
                return 3
            elif row['dx3'] == rc_diag_values[count]:
                return 3
            elif row['dx4'] == rc_diag_values[count]:
                return 3
            elif row['dx5'] == rc_diag_values[count]:
                return 3
            elif row['dx6'] == rc_diag_values[count]:
                return 3
            elif row['dx7'] == rc_diag_values[count]:
                return 3
            elif row['dx8'] == rc_diag_values[count]:
                return 3
            elif row['dx9'] == rc_diag_values[count]:
                return 3
            elif row['dx10'] == rc_diag_values[count]:
                return 3
            elif row ['dx11'] == rc_diag_values[count]:
                return 3
            return float(0)
        joined_db_1[syndrome_names[count]] = joined_db_1.apply(lambda row: categorise(row), axis = 1)
        count += 1
    
    ######## Creating ADHD DB ###########################
    adhd_pt_1 = joined_db_1[['ADHD-inattentive', 'ADHD-hyperactive-imoulsive', 'ADHD-combined', 'ADHD Inattention', 'ADHD Hyperactivity', 'ADHD Diagnosis']]

    adhd_pt_1['ADHD Inattentive Diagnosis'] = np.where((((adhd_pt_1['ADHD Inattention'] == 3) & (adhd_pt_1['ADHD Diagnosis'] == 3)) & (adhd_pt_1['ADHD Hyperactivity'] != 3)), 3, 0)
    adhd_pt_1['ADHD Hyperactive Diagnosis'] = np.where((((adhd_pt_1['ADHD Hyperactivity'] == 3) & (adhd_pt_1['ADHD Diagnosis'] == 3)) & (adhd_pt_1['ADHD Inattention'] != 3)), 3, 0)
    adhd_pt_1['ADHD Combined Diagnosis'] = np.where(((adhd_pt_1['ADHD Hyperactivity'] == 3) & (adhd_pt_1['ADHD Inattention'] == 3) & (adhd_pt_1['ADHD Diagnosis'] == 3)), 3, 0)

    adhd_db = adhd_pt_1.drop(columns = ['ADHD-inattentive', 'ADHD-hyperactive-imoulsive', 'ADHD-combined', 'ADHD Inattention', 'ADHD Hyperactivity', 'ADHD Diagnosis'], axis = 1)

    ############ Creatin the Joined DB (pt. 2) ###################
    joined_db = pd.concat([joined_db_1, adhd_db], axis = 1)

    # Reorerding columns so that ADHD diagnoses are not with the PI information.
    joined_db = joined_db[['dx1', 'dx2', 'dx3', 'dx4', 'dx5', 'dx6', 'dx7', 'dx8', 'dx9', 'dx10', 'dx11', 'CMDE Diagnosis', 'PMDE Diagnosis', 'CME Diagnosis',
    'CME Alt Diagnosis', 'CHME Diagnosis', 'CHME Alt Diagnosis', 'PME Diagnosis', 'PME Alt Diagnosis', 'PHME Diagnosis', 'PHME Alt Diagnosis','CCD Diagnosis', 
    'PDD Diagnosis', 'OSFED Diagnosis', 'ADHD Diagnosis', 'ASD Diagnosis', 'AD Diagnosis', 'PPD Diagnosis','SPD Diagnosis', 'SDPD Diagnosis', 'BPD Diagnosis', 
    'APD Diagnosis', 'Schizophrenia Diagnosis', 'Schizophreniform Diagnosis', 'SchizoAffective Diagnosis','BID Diagnosis', 'BIID Diagnosis', 'MDD Diagnosis', 
    'BrPD Diagnosis', 'Delusional Disorder Diagnosis', 'OSAD Diagnosis', 'OSOCD Diagnosis', 'ADHD Inattention','ADHD Hyperactivity', 'ADHD Inattentive Diagnosis', 
    'ADHD Hyperactive Diagnosis', 'ADHD Combined Diagnosis', 'OSTSRD Diagnosis', 'PDD Lifetime Diagnosis','AUD Lifetime Diagnosis', 'SUD Sed/Hyp/Anx Lifetime Diagnosis', 
    'SUD Can Lifetime Diagnosis', 'SUD Stim Lifetime Diagnosis', 'SUD Opi Lifetime Diagnosis','SUD Inh Lifetime Diagnosis', 'SUD PCP Lifetime Diagnosis', 
    'SUD Hall Lifetime Diagnosis', 'SUD Other Lifetime Diagnosis', 'PD Lifetime Diagnosis','AGORA Lifetime Diagnosis', 'SAD Lifetime Diagnosis', 'SP Lifetime Diagnosis', 
    'GAD Lifetime Diagnosis', 'OCD Lifetime Diagnosis', 'AN Lifetime Diagnosis','BN Lifetime Diagnosis', 'BE Lifetime Diagnosis', 'PTSD Lifetime Diagnosis', 
    'Major Depressive Disorder', 'Bipolar I Disorder', 'Bipolar II Disorder','Cyclothymia', 'Persistent Depressive Disorder', 'Prementral Dsyphoric Disorder', 
    'Other Mood Disorder', 'Schizophrenia', 'Schizoaffective Disorder','Schizophreniform Disorder', 'Brief Psychotic Disorder', 'Delusional Disorder', 'Other Psychotic Disorder', 
    'Alcohol Use Disorder', 'Sedative Hypnotic, or Anciolytic Use Disorder','Cannabis Use Disorder', 'Stimulant Use Disorder', 'Opiate Use Disorder', 'Inhalant Use Disorder',
    'Phencyclidine and related substances Use Disorder', 'Hallucinogen Use Disorder', 'Other Substance Use Disorder', 'Panic Disorder', 'Agoraphobia', 'Social Anxiety Disorder', 
    'Specific Phobia', 'Generalized Anxiety Disorder','Other Anxiety Disorder', 'Obsessive Compulsive Disorder', 'Other OCD Disorder', 'Anorexia Nervosa', 'Bulimia Nervosa', 
    'Binge Eating Disorder', 'Other Eating Disorder','ADHD-inattentive', 'ADHD-hyperactive-imoulsive', 'ADHD-combined', 'Acute Stress Disorder', 'Posttraumatic Stress Disorder', 
    'Adjustment Disorder', 'Other Trauma and Stress Related Disorder','Paranoid Personality Disorder', 'Schizotypal Personality Disorder', 'Schizoid Personality Disorder', 
    'Borderline Personality Disorder', 'Anti-social Personalisty Disorder','Healthy Control', 'Excluded']]

    ############################################################## Creating the comparison columns #############################################################################################

    # Looping through the data frame to create comparison columns
    count_new = 0
    count_easy = 0
    complex_count = 0
    while count_new < len(syndrome_names):
        if syndrome_names[count_new] in cleaned_complex_syndrome_list:
            joined_db[cleaned_complex_syndrome_list[complex_count] + " Comparison"] = np.where(((joined_db[cleaned_complex_syndrome_list[complex_count]] == 
            joined_db[lifetime_column_names_final[complex_count]])), "Identical", "Different")
            count_new += 1
            complex_count += 1
        elif syndrome_names[count_new] in cleaned_easy_syndrome_list:
            joined_db[cleaned_easy_syndrome_list[count_easy] + " Comparison"] = np.where((joined_db[cleaned_easy_syndrome_list[count_easy]] == 
            joined_db[cleaned_non_na_vumc_syndromes[count_easy]]), "Identical", "Different")
            count_new += 1
            count_easy += 1
        else:
            count_new += 1

    # Creating columns that indicat direction of discrepancy
    countt = 0
    countt_easy = 0
    countt_complex = 0
    while countt < len(syndrome_names):
        if syndrome_names[countt] in cleaned_complex_syndrome_list:
            joined_db[cleaned_complex_syndrome_list[countt_complex] + " Comparison Direction"] = np.where(((joined_db[cleaned_complex_syndrome_list[countt_complex]] == 3) &
            (joined_db[lifetime_column_names_final[countt_complex]] == 0)), "SCID", np.where(((joined_db[cleaned_complex_syndrome_list[countt_complex]] == 0) &
            (joined_db[lifetime_column_names_final[countt_complex]] == 3)), "Patient Information", "No Problem"))
            countt += 1
            countt_complex += 1
        elif syndrome_names[countt] in cleaned_easy_syndrome_list:
            joined_db[cleaned_easy_syndrome_list[countt_easy] + " Comparison Direction"] = np.where((joined_db[cleaned_easy_syndrome_list[countt_easy]] == 3) & 
            (joined_db[cleaned_non_na_vumc_syndromes[countt_easy]] == 0), "SCID", np.where(((joined_db[cleaned_easy_syndrome_list[countt_easy]] == 0) & 
            (joined_db[cleaned_non_na_vumc_syndromes[countt_easy]] == 3)), "Patient Information", "No Problem"))
            countt += 1
            countt_easy += 1
        else:
            countt += 1

    ######################################################### Getting the subject IDs associated with discrepancies #############################################################################

    # Getting a list of the comparison columns
    comparison_db = joined_db.filter(regex= "Comparison$", axis = 1)
    comparison_columns = comparison_db.columns.tolist()

    # While loop to create a list of lists with the record ids of each syndrome
    comparison_count = 0
    Master_list = []
    while comparison_count < len(comparison_columns):
        needed_ids = comparison_db.index[comparison_db[comparison_columns[comparison_count]] == 'Different'].tolist()
        Master_list.append(needed_ids)
        comparison_count += 1

    # Because not each syndrome has the same number of IDs we need to equalize the lengths in order to form a dataframe
    maxLen = max(len(item) for item in Master_list)
    for item in Master_list:
        while len(item) < maxLen:
            item.append(None)

    master_dictionary = dict(zip(comparison_columns, Master_list))

    master_db = pd.DataFrame.from_dict(master_dictionary)
else:
    st.write('Please Upload a Datafile.')

############################################################################### Dashboard Components #######################################################################################
@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8') 

if data_file:
    see_data = st.sidebar.checkbox('Individual Data Files')
    st.markdown('## Dashboard Uses')
    st.markdown('***')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('If you select the "Individual Data Files" box on the left, you will be presented with five tabs that show the following data:')
        st.markdown("- Raw Data")
        st.markdown("- Filtered Data (Only subjects with the SCID completed and not excluded.)")
        st.markdown("- VUMC Crosswalk (Looking at the module diagnosis items)")
        st.markdown("- RC Crosswalk (Looking at the Patient Information diagnoses items)")
        st.markdown("- Comparison Dataset (The actual dataset used to discover the specific discrepancies)")
        st.markdown("- Discrepant IDs (List of IDs associated with specific syndromes")
        st.markdown('''
        <style>
        [data-testid="stMarkdownContainer"] ul{
            list-style-position: inside;
        }
        <style>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown('If you select the "Specific Disorder" box you will have the option to look at the subject IDs that have discrepancies for specific disorders. However, there are some disorders missing from this preliminary analysis which are listed below.')
        st.markdown('- Other Mood Disorder')
        st.markdown('''
        <style>
        [data-testid="stMarkdownContainer"] ul{
            list-style-position: inside;
        }
        <style>
        ''', unsafe_allow_html=True)
    if see_data:
        st.markdown('## Data Outputs')
        st.markdown('***')
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Raw Data', 'Filtered Data', 'VUMC Crosswalk', 'RC Crosswalk', 'Comparison Dataset', 'Discrepant IDs'])
        with tab1:
            st.subheader('Raw Data')
            st.write('This is the unflitered raw data. Based of the 11/29 VUMC Redcap.')
            st.markdown('**Helpful tip:** if you want to search the DB use cmd + f if using a Mac and ctrl + f if using windows!')
            st.write(full_db)
            st.write('The number of subjects is', len(full_db.index))
            csv = convert_df(full_db)
            st.download_button(label= 'Download Data as a CSV', data = csv, file_name = 'full_rc_11_29_dataset.csv', mime= 'text/csv')
        with tab2:
            st.subheader('Filtered Data')
            st.write('This data was filtered using the parameters "Scid_iscomplete" == "yes" and "dx1" is not 47 (Excluded).')
            st.markdown('**Helpful tip:** if you want to search the DB use cmd + f if using a Mac and ctrl + f if using windows!')
            st.write(filtered_full_db)
            st.write("The numbers of subjects is", len(filtered_full_db.index))
            csv = convert_df(filtered_full_db)
            st.download_button(label= 'Download Data as a CSV', data = csv, file_name = 'filtered_rc_11_29_dataset.csv', mime= 'text/csv')
        with tab3:
            st.subheader('VUMC Crosswalk')
            st.markdown('**Helpful tip:** if you want to search the DB use cmd + f if using a Mac and ctrl + f if using windows!')
            st.checkbox('Use Page Width', value = False, key = 'use_container_width')
            st.dataframe(vumc_item_syn_db.iloc[:, 0:2], use_container_width=st.session_state.use_container_width)
            csv = convert_df(vumc_item_syn_db)
            st.download_button(label= 'Download Data as a CSV', data = csv, file_name = 'VUMC_crosswalk.csv', mime= 'text/csv')
        with tab4:
            st.subheader('RC Crosswalk')
            st.markdown('**Helpful tip:** if you want to search the DB use cmd + f if using a Mac and ctrl + f if using windows!')
            st.checkbox('Use Page Width', value = False, key = 'use_container_widths')
            st.dataframe(crosswalk_db.iloc[:, 0:4], use_container_width=st.session_state.use_container_widths)
            csv = convert_df(crosswalk_db)
            st.download_button(label= 'Download Data as a CSV', data = csv, file_name = 'RC_crosswalk.csv', mime= 'text/csv')
        with tab5:
            st.subheader('Comparison Dataset')
            st.write('This is the fully filtered dataset used to perform the comparative analysis. This dataset includes all of the dx columns from the Patient Information Page, the Module Diagnoses, and the results of the comparisons in their own columns.')
            st.markdown('**Rule of thumb:** The columns that have Diagnosis at the end are from the SCID and the ones without diagnosis at the end are from the Patient Information Page.')
            st.markdown('**Helpful tip:** if you want to search the DB use cmd + f if using a Mac and ctrl + f if using windows!')
            only_ids = st.checkbox('Only Discrepanct IDs?')
            interviewers_included = st.checkbox('See Associated Interviewers?')
            if only_ids:
                count = 0
                subject_id_list = []
                columns = master_db.columns.tolist()
                while count < len(columns):
                    subject_id_list.append(master_db[columns[count]].values)
                    count += 1
                subject_id_list_1 = [x for x in subject_id_list if str(x) != None]
                flat_list = [item for sublist in subject_id_list_1 for item in sublist]
                cleaned_flat_list = [x for x in flat_list if x != None]
                only_wanted_ids = []
                for x in cleaned_flat_list:
                    if x not in only_wanted_ids:
                        only_wanted_ids.append(x)
                data = joined_db.loc[only_wanted_ids, :]
                st.write('Only subjects who have discrepancies')
                st.dataframe(data.iloc[:, 21:])
                st.write("Number of discrepant Subjects:", len(data.index))
                csv = convert_df(data.iloc[:, 21:])
                st.download_button(label = 'Download Data as a CSV', data = csv, file_name = 'VUMC_Discrepant_ids.csv', mime = 'text/csv')
            else:
                st.dataframe(joined_db.iloc[:, 21:])
                csv = convert_df(joined_db.iloc[:, 21:])
                st.download_button(label= 'Download Data as a CSV', data = csv, file_name = 'Full_joined_db.csv', mime= 'text/csv')
            if interviewers_included:
                if only_ids:
                    count = 0
                    subject_id_list = []
                    columns = master_db.columns.tolist()
                    while count < len(columns):
                        subject_id_list.append(master_db[columns[count]].values)
                        count += 1
                    subject_id_list_1 = [x for x in subject_id_list if str(x) != None]
                    flat_list = [item for sublist in subject_id_list_1 for item in sublist]
                    cleaned_flat_list = [x for x in flat_list if x != None]
                    only_wanted_ids = []
                    for x in cleaned_flat_list:
                        if x not in only_wanted_ids:
                            only_wanted_ids.append(x)
                    Interviewer_data = filtered_full_db.loc[only_wanted_ids, "scid_interviewername"]
                    refined_data_2 = joined_db.loc[only_wanted_ids, :]
                    data = pd.concat([Interviewer_data, refined_data_2.iloc[:, 21:]], axis=1)
                    st.write('Only subjects with discrepancies and their corresponding interviewers.')
                    st.dataframe(data)
                    csv = convert_df(data)
                    st.download_button(label = 'Download Data as a CSV', data = csv, file_name = 'VUMC_Discrepant_ids_and_ints.csv', mime = 'text/csv')
        with tab6:
            st.subheader('Discrepant IDs')
            st.write('This file contains a list of the subject IDs associated with discrepancies found in select syndromes. Notably one disorder was left out as there is no direct diagnosis in the interview. The syndrome is listed below.')
            st.write('- Other Mood Disorder.')
            st.markdown('**Helpful tip:** if you want to search the DB use cmd + f if using a Mac and ctrl + f if using windows!')
            st.write(master_db)
            csv = convert_df(master_db)
            st.download_button(label = 'Download Data as a CSV', data = csv, file_name = 'Discrepant_ids.csv', mime= 'text/csv')
            st.write('Table below shows the number of subjects per disorder. If you click on the 0 you can order the list by # of subjects.')
            st.write(master_db.count())
    specific_disorder = st.sidebar.checkbox('Specific Disorder')
    specific_disorder_names = master_db.columns.tolist()
    count_checkbox = 0
    master_db_columns = []
    while count_checkbox < len(specific_disorder_names):
        syndrome_name_stuff = specific_disorder_names[count_checkbox]
        syndrome_name = syndrome_name_stuff.removesuffix(' Comparison')
        master_db_columns.append(syndrome_name)
        count_checkbox += 1

    lifetime_dash_list = vumc_item_syn_db.lifetime_names.tolist()
    cleaned_lifetime_dash_list = [x for x in lifetime_dash_list if str(x) != 'nan']
    normal_dash_list = vumc_item_syn_db.full_names.tolist()
    cleaned_normal_dash_list = [x for x in normal_dash_list if str(x) != 'nan']
    dash_dict = {normal_dash_list[i]: cleaned_lifetime_dash_list[i] for i in range(len(cleaned_lifetime_dash_list))}

    dash_easy_dict = {cleaned_easy_syndrome_list[i]: cleaned_non_na_vumc_syndromes[i] for i in range(len(cleaned_easy_syndrome_list))}
    adhd_list = cleaned_easy_syndrome_list[-3:]

    # Discrepancy Direction
    direction_db = joined_db.filter(regex='Direction$', axis = 1)
    direction_columns = direction_db.columns.tolist()

    if specific_disorder:
        filter_type = st.sidebar.selectbox('How would you like to filter data', ['', 'Specific Disorder', 'Interviewer', 'Subject_id'])
        if filter_type == '':
            st.markdown('---')
            st.markdown('## Please choose a filter criteria')
            st.markdown('Options:')
            st.markdown('- Specific Disorder - Filter by specific disorders.')
            st.markdown('- Interviewer - Filter by interviewer.')
            st.markdown('- Subject_id - Filter by subject ID.')
        elif filter_type == 'Specific Disorder':
            final_options = st.sidebar.selectbox('Which Syndrome would you like to look at?', master_db_columns)
            for i in range(44):
                    if final_options == master_db_columns[i]:
                        st.markdown('# Data Outputs')
                        st.markdown('***')
                        st.markdown(f'## {master_db_columns[i]}')
                        st.markdown(f'Table of Subject IDs that have a discrepancy with **{master_db_columns[i]}**')
                        data = master_db[specific_disorder_names[i]].dropna(axis = 0)
                        st.write(data)
                        st.write("The number of discrepant subjects is", len(data.index))
                        csv = convert_df(data)
                        st.download_button(label = 'Download Data as a CSV', data = csv, file_name = f'{master_db_columns[i]} ids.csv', mime= 'text/csv')
            interviewer = st.sidebar.checkbox('See Interviewer?')
            specific_items = st.sidebar.checkbox('See the Specific discrepancy?')
            if specific_items:
                if interviewer:
                    for i in range(44):
                        if final_options == master_db_columns[i]:
                            if master_db_columns[i] in adhd_list:
                                lifetime_column = dash_easy_dict.get(master_db_columns[i])
                                refined_db = master_db[specific_disorder_names[i]].dropna(axis = 0)
                                subject_id_list = refined_db.tolist()
                                interviewer_columns = filtered_full_db.loc[subject_id_list, 'scid_interviewername']
                                desired_data = joined_db.loc[subject_id_list, [lifetime_column, *adhd_list, direction_columns[i]]]
                                final_data = pd.concat([interviewer_columns, desired_data], axis =1)
                                st.write('Table below shows first the diagnosis found within the module and second column is from the Patient information page.')
                                st.write('The ADHD diagnosis column on the far left was created using scid_k29 (Diagnosis) and either scid_k13 (hyperactive threshold) or scid_k23 (inattentive threhsold).')
                                st.write('''Direction columns indicates the direction of the discrepancy. If Patient Information is listed that means that a diagnosis
                                was found in the SCID but not on the PI. The reverse is true if SCID is listed.''')
                                st.write(final_data)
                                csv = convert_df(final_data)
                                st.download_button(label = 'Download Data as a CSV', data = csv, file_name = f'{master_db_columns[i]} ids and discrepancies.csv', mime= 'text/csv')
                                st.write("Interviewer Count:")
                                interviewer_columns = interviewer_columns.reset_index()
                                st.write(interviewer_columns['scid_interviewername'].value_counts())
                            elif master_db_columns[i] in normal_dash_list:
                                lifetime_column = dash_dict.get(master_db_columns[i])
                                refined_db = master_db[specific_disorder_names[i]].dropna(axis = 0)
                                subject_id_list = refined_db.tolist()
                                interviewer_columns = filtered_full_db.loc[subject_id_list, 'scid_interviewername']
                                desired_data = joined_db.loc[subject_id_list, [lifetime_column, master_db_columns[i], direction_columns[i]]]
                                final_data = pd.concat([interviewer_columns, desired_data], axis =1)
                                st.write('Table below shows first the "Lifetime Diagnosis" which is the diagnosis found within the module and second the name of the disorder which is the information from the Patient information page.')
                                st.write(final_data)
                                csv = convert_df(final_data)
                                st.download_button(label = 'Download Data as a CSV', data = csv, file_name = f'{master_db_columns[i]} ids and discrepancies.csv', mime= 'text/csv')
                                st.write("Interviewer Count:")
                                interviewer_columns = interviewer_columns.reset_index()
                                st.write(interviewer_columns['scid_interviewername'].value_counts())
                            else:
                                vumc_column = dash_easy_dict.get(master_db_columns[i])
                                refined_db = master_db[specific_disorder_names[i]].dropna(axis = 0)
                                subject_id_list = refined_db.tolist()
                                interviewer_columns = filtered_full_db.loc[subject_id_list, 'scid_interviewername']
                                desired_data = joined_db.loc[subject_id_list, [vumc_column, master_db_columns[i], direction_columns[i]]]
                                final_data = pd.concat([interviewer_columns, desired_data], axis =1)
                                st.write('Table below shows first the diagnosis found within the module and second the name of the disorder which was found in the Patient Information page.')
                                st.write('The actual scid variable name can be found in the VUMC Crosswalk which you can look at if you click the "Individual Data Files" Checkbox.')
                                st.write(final_data)
                                csv = convert_df(final_data)
                                st.download_button(label = 'Download Data as a CSV', data = csv, file_name = f'{master_db_columns[i]} ids and discrepancies.csv', mime= 'text/csv')
                                st.write("Interviewer Count:")
                                interviewer_columns = interviewer_columns.reset_index()
                                st.write(interviewer_columns['scid_interviewername'].value_counts())
                else:
                    for i in range(44):
                        if final_options == master_db_columns[i]:
                            if master_db_columns[i] in adhd_list:
                                lifetime_column = dash_easy_dict.get(master_db_columns[i])
                                refined_db = master_db[specific_disorder_names[i]].dropna(axis = 0)
                                subject_id_list = refined_db.tolist()
                                desired_data = joined_db.loc[subject_id_list, [lifetime_column, *adhd_list, direction_columns[i]]]
                                st.write('Table below shows first the diagnosis found within the module and second column is from the Patient information page.')
                                st.write('The ADHD diagnosis column on the far left was created using scid_k29 (Diagnosis) and either scid_k13 (hyperactive threshold) or scid_k23 (inattentive threhsold).')
                                st.write('''Direction columns indicates the direction of the discrepancy. If Patient Information is listed that means that a diagnosis
                                was found in the SCID but not on the PI. The reverse is true if SCID is listed.''')
                                st.write(desired_data)
                                csv = convert_df(desired_data)
                                st.download_button(label = 'Download Data as a CSV', data = csv, file_name = f'{master_db_columns[i]} ids and discrepancies.csv', mime= 'text/csv')
                            elif master_db_columns[i] in normal_dash_list:
                                lifetime_column = dash_dict.get(master_db_columns[i])
                                refined_db = master_db[specific_disorder_names[i]].dropna(axis = 0)
                                subject_id_list = refined_db.tolist()
                                desired_data = joined_db.loc[subject_id_list, [lifetime_column, master_db_columns[i], direction_columns[i]]]
                                st.write('Table below shows first the "Lifetime Diagnosis" which is the diagnosis found within the module and second the name of the disorder which is the information from the Patient information page.')
                                st.write(desired_data)
                                csv = convert_df(desired_data)
                                st.download_button(label = 'Download Data as a CSV', data = csv, file_name = f'{master_db_columns[i]} ids and discrepancies.csv', mime= 'text/csv')
                            else:
                                vumc_column = dash_easy_dict.get(master_db_columns[i])
                                refined_db = master_db[specific_disorder_names[i]].dropna(axis = 0)
                                subject_id_list = refined_db.tolist()
                                desired_data = joined_db.loc[subject_id_list, [vumc_column, master_db_columns[i], direction_columns[i]]]
                                st.write('Table below shows first the diagnosis found within the module and second the name of the disorder which was found in the Patient Information page.')
                                st.write('The actual scid variable name can be found in the VUMC Crosswalk which you can look at if you click the "Individual Data Files" Checkbox.')
                                st.write(desired_data)
                                csv = convert_df(desired_data)
                                st.download_button(label = 'Download Data as a CSV', data = csv, file_name = f'{master_db_columns[i]} ids and discrepancies.csv', mime= 'text/csv')
            if not specific_items:
                if interviewer:
                    for i in range (44):
                        if final_options == master_db_columns[i]:
                            refined_data = master_db[specific_disorder_names[i]].dropna(axis = 0)
                            column_list = refined_data.tolist()
                            interview_list = filtered_full_db.loc[column_list, 'scid_interviewername']
                            interview_list = interview_list.reset_index()
                            st.write('Table of subjects and their interviewer:')
                            st.write(interview_list)
                            csv = convert_df(interview_list)
                            st.download_button(label = 'Download Data as a CSV', data = csv, file_name = f'{master_db_columns[i]} ids and interviewers.csv', mime= 'text/csv')
                            st.write("Interviewer Count:")
                            st.write(interview_list['scid_interviewername'].value_counts())
        elif filter_type == 'Interviewer':
            count = 0
            subject_id_list = []
            columns = master_db.columns.tolist()
            while count < len(columns):
                subject_id_list.append(master_db[columns[count]].values)
                count += 1
            subject_id_list_1 = [x for x in subject_id_list if str(x) != None]
            flat_list = [item for sublist in subject_id_list_1 for item in sublist]
            cleaned_flat_list = [x for x in flat_list if x != None]
            only_wanted_ids = []
            for x in cleaned_flat_list:
                if x not in only_wanted_ids:
                    only_wanted_ids.append(x)
            Interviewer_data = filtered_full_db.loc[only_wanted_ids, "scid_interviewername"]
            refined_data_2 = joined_db.loc[only_wanted_ids, :]
            data = pd.concat([Interviewer_data, refined_data_2.iloc[:, 21:]], axis=1)
            interviewer_list = data['scid_interviewername'].unique()
            specific_interviewer = st.sidebar.selectbox('Which interviewer would you like to look like?', ['', *interviewer_list])
            specific_disorders = st.sidebar.checkbox('Look at Specific Disorders')
            if specific_interviewer == '':
                st.markdown('---')
                st.markdown('## Please choose an interviewer to get started.')
                interviewer_num = []
                for x in interviewer_list:
                    number = data[data['scid_interviewername'] == x].index
                    interviewer_num.append(len(number))
                int_num_dict = {interviewer_list[i]:interviewer_num[i] for i in range(len(interviewer_list))}
                int_num_db = pd.DataFrame.from_dict(int_num_dict, orient= 'index')
                st.write(int_num_db)
            elif specific_interviewer != '':
                for i in range(len(interviewer_list)):
                    if specific_interviewer == interviewer_list[i]:
                        st.markdown('# Data Outputs')
                        st.markdown('***')
                        col1, col2 = st.columns(2)
                        with col1:
                            name = str(interviewer_list[i])
                            name = name.split(',')
                            name = name[1]
                            desired_db = data[data['scid_interviewername'] == interviewer_list[i]]
                            st.markdown(f'List of Discrepant Subjects for {name} :')
                            st.write(desired_db.index)
                            st.write('Number of Discrepant Subjects:', len(desired_db.index))
                        if specific_disorders:
                            with col2:
                                column_list_spec = desired_db.columns.tolist()
                                desired_column_list = column_list_spec[96:140]
                                disorder_list = []
                                filter_db = desired_db.loc[:, desired_column_list]
                                Disorders = filter_db.apply(lambda row: row[row != 'Identical'].index.tolist(), axis =1)
                                new_Disorders = Disorders.to_frame()
                                new_Disorders = new_Disorders.rename(columns = {0:'Discrepant Syndromes'})
                                st.write('Table of discrepant subjects and which syndromes are discrepant.')
                                st.write(new_Disorders)
                                csv = convert_df(new_Disorders)
                                st.download_button(label = 'Download Data as a CSV', data = csv, file_name = f'{interviewer_list[i]} subject discrepancies.csv', mime= 'text/csv')

