import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import locale
from matplotlib.dates import DateFormatter
warnings.filterwarnings("ignore")
mpl.rc('font',family='Times New Roman') #везде TNR
locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8') #русский язык
formatter = DateFormatter('%B \n %Y г.') #формат даты

class URussFact:
    def __init__(self):
        self.plot_builder = PlotBuilder() 
        self.class_gdh = GetSpch()

    def get_data(self, path):
        os.makedirs('Name', exist_ok=True)
        folders = Path(path)
        file_list = list(folders.rglob('*.xlsx'))

        for files in file_list:
            files_1 = files + '\\' + 'Name_file_1.xlsx'
            Excel_1 = pd.read_excel(files_1)
            Excel_1.replace({0:np.nan}, inplace=True)
            Excel_1.replace({'РЕМ':np.nan}, inplace=True)
            Excel_1.replace({'РЕЗ':np.nan}, inplace=True)
            Excel_1.replace({' ':np.nan}, inplace=True)

            bedin = Excel_1[Excel_1.iloc[:,0] == 'Дата'].index[0]
            end = Excel_1[Excel_1.iloc[:,0] == 'Итого'].index[0]
            df_new = Excel_1.iloc[bedin+3:end, :]
            df_itog = Excel_1.iloc[end, :]
            df_new['Unnamed: 0'] = pd.to_datetime(df_new.iloc[:,0])
            df_new['Unnamed: 0'] = df_new['Unnamed: 0'].dt.strftime('%d.%m.%Y')
            date = df_new.iloc[:,0]

            #год
            p_in_1st = (df_new.iloc[:,[5,12,19,26,33,40,47]] * 0.0980665 + 0.0980665).mean(axis=1)
            st_sg_1st = df_new.iloc[:,52] 
            p_out_2st = (df_new.iloc[:,[64,71,78,85,92,99,106]] * 0.0980665 + 0.0980665).mean(axis=1)
            st_sg_2st = df_new.iloc[:,109] 
            #месяц
            date_m = pd.Series(df_new.iloc[0,0])
            p_in_1st_m = np.nanmean((df_itog.iloc[[5,12,19,26,33,40,47]] * 0.0980665 + 0.0980665).values)
            t_in_1st_m = np.nanmean(df_itog.iloc[[6,13,20,27,34,41,48]].values)
            p_out_1st_m = np.nanmean((df_itog.iloc[[7,14,21,28,35,42,49]] * 0.0980665 + 0.0980665).values)
            t_out_1st_m = np.nanmean(df_itog.iloc[[8,15,22,29,36,43,50]].values)
            st_sg_1st_m = df_itog.iloc[52] 
            obor_1st = np.nanmean(df_itog.iloc[[3,10,17,24,31,38,45]].values)
            p_avo_1st = (df_new.iloc[:,55].loc[df_new.iloc[:,55] > 10] * 0.0980665 + 0.0980665).mean()
            t_avo_1st = df_itog.iloc[56]
            W_a_1_st = np.mean([elem for elem in df_new.iloc[:,[5,12,19,26,33,40,47]].count(axis=1) if elem !=0]).__round__(0) 
            p_in_2st_m = np.nanmean((df_itog.iloc[[62,69,76,83,90,97,104]] * 0.0980665 + 0.0980665).values)
            t_in_2st_m = np.nanmean(df_itog.iloc[[63,70,77,84,91,98,105]].values)
            p_out_2st_m = np.nanmean((df_itog.iloc[[64,71,78,85,92,99,106]] * 0.0980665 + 0.0980665).values)
            t_out_2st_m = np.nanmean(df_itog.iloc[[65,72,79,86,93,100,107]].values)
            st_sg_2st_m = df_itog.iloc[109] 
            obor_2st = np.nanmean(df_itog.iloc[[60,67,74,81,88,95,102]].values)
            p_avo_2st = (df_new.iloc[:,110].loc[df_new.iloc[:,110] > 30] * 0.0980665 + 0.0980665).mean()
            t_avo_2st = (df_new.iloc[:,111].loc[df_new.iloc[:,111] > 0]).mean()
            W_a_2_st = np.mean([elem for elem in df_new.iloc[:,[62,69,76,83,90,97,104]].count(axis=1) if elem !=0]).__round__(0) 
            t_tepl = df_itog.iloc[117]
            p_in_tepl = df_itog.iloc[114] * 0.0980665 + 0.0980665
            p_out_tepl_2 = df_itog.iloc[116] * 0.0980665 + 0.0980665

            files_2 = files + '\\' + 'Name_file_2.xlsx'
            Excel_2 = pd.read_excel(files_2)
            Excel_2.replace({0:np.nan}, inplace=True)
            bedin = Excel_2[Excel_2.iloc[:,0] == 'Дата'].index[0]
            end = Excel_2[Excel_2.iloc[:,0] == 'Итого'].index[0]
            df_new_2 = Excel_2.iloc[bedin+4:end, :]
            df_itog_2 = Excel_2.iloc[end, :]
            Q_g = df_new_2.iloc[:, 15] / 1000
            Q_g_m = Q_g.mean()
            Ttr = df_new_2.iloc[:, 5] 
            p_out_uzg = df_itog_2.iloc[3] * 0.0980665 + 0.0980665

            files_3 = files + '\\' + 'Name_file_3.xlsx'
            Excel_3 = pd.read_excel(files_3)
            Excel_3.replace({0:np.nan}, inplace=True)
            Excel_3.replace({'РЕМ':np.nan}, inplace=True)
            Excel_3.replace({'РЕЗ':np.nan}, inplace=True)
            bedin = Excel_3[Excel_3.iloc[:,0] == 'Дата'].index[0]
            end = Excel_3[Excel_3.iloc[:,0] == 'Итого'].index[0]
            df_new_3 = Excel_3.iloc[bedin+4:end, :]
            df_itog_3 = Excel_3.iloc[end, :]
            p_in_ppa = df_new_3.iloc[:,1] * 0.0980665 + 0.0980665
            p_in_ppa_m = df_itog_3.iloc[1] * 0.0980665 + 0.0980665
            p_in_kgs = df_itog_3.iloc[29] * 0.0980665 + 0.0980665
            t_in_kgs = df_itog_3.iloc[30]
            p_out_kgs = np.nanmean((df_itog_3.iloc[[31,34,37,40,43,46,49,52]] * 0.0980665 + 0.0980665).values)
            t_out_kgs = np.nanmean((df_itog_3.iloc[[32,35,38,41,44,47,50,53]]).values)
            W_a_kgs =  np.mean([elem for elem in df_new_3.iloc[:,[31,34,37,40,43,46,49,52]].count(axis=1) if elem !=0]).__round__(0) 
            t_kgs_tepl = df_itog.iloc[2]
            Q_1kgs = Q_g_m / W_a_kgs
            dp_ppa = abs(p_in_ppa_m - p_in_kgs)
            dp_sep = abs(p_in_kgs - p_out_kgs)
            p_out_tepl = df_itog.iloc[1] * 0.0980665 + 0.0980665
            dp_tepl = abs(p_out_kgs - p_out_tepl)
            dp_tepl_1st = abs(p_out_tepl - p_in_1st_m)
            p_nagn = df_itog.iloc[53] * 0.0980665 + 0.0980665
            dp_1st_nagn = abs(p_out_1st_m - p_nagn)
            dp_nagn_avo_1 = abs(p_nagn - p_avo_1st)
            dp_avo_1_2st = abs(p_avo_1st - p_in_2st_m)
            dp_all_ppa_gpa = np.sum([dp_ppa, dp_sep, dp_tepl, dp_tepl_1st, dp_1st_nagn, dp_nagn_avo_1, dp_avo_1_2st])
            dp_avo_2_tepl = abs(p_avo_2st - p_out_2st_m)
            dp_tepl_2 = abs(p_in_tepl - p_out_tepl_2)
            dp_all_gpa_tepl = np.sum([dp_avo_2_tepl, dp_tepl_2])

            files_4 = files + '\\' + 'Name_file_4.xlsx'
            Excel_4 = pd.read_excel(files_4)
            end = Excel_4[Excel_4.iloc[:,0] == 'Всего\nза месяц'].index[0]
            df_itog_4 = Excel_4.iloc[end, :]
            BMC_kgs = df_itog_4.iloc[2]    

            files_5 = files + '\\' + 'Name_file_5.xlsx'
            Excel_5 = pd.read_excel(files_5)
            Excel_5.replace({0:np.nan}, inplace=True)
            Excel_5.replace({'РЕМ':np.nan}, inplace=True)
            Excel_5.replace({'РЕЗ':np.nan}, inplace=True)
            Excel_5.replace({' ':np.nan}, inplace=True)
            bedin = Excel_5[Excel_5.iloc[:,0] == 'Дата'].index[0]
            end = Excel_5[Excel_5.iloc[:,0] == 'Итого'].index[0]
            df_new_5 = Excel_5.iloc[bedin+3:end, :]
            df_itog_5 = Excel_5.iloc[end, :]
            p_in_abs = df_itog.iloc[116] * 0.0980665 + 0.0980665
            p_out_abs = np.nanmean((df_itog_5.iloc[[1,4,7,10,13,16,19,22]] * 0.0980665 + 0.0980665).values)
            t_abs = np.nanmean((df_itog_5.iloc[[2,5,8,11,14,17,20,23]]).values)
            W_a_abs =  np.mean([elem for elem in df_new_5.iloc[:,[1,4,7,10,13,16,19,22]].count(axis=1) if elem !=0]).__round__(0) 
            Q_1abs = Q_g_m / W_a_abs
            Ttr_m = np.nanmean(df_new_2.iloc[:, 5]) 
            dp_tepl_abs = abs(p_out_tepl_2 - p_out_abs)
            dp_abs_uzg = abs(p_out_abs - p_out_uzg)
            dp_all_tepl_uzg = np.sum([dp_tepl_abs,dp_abs_uzg])
            dp_all = np.sum([dp_all_ppa_gpa,dp_all_gpa_tepl,dp_all_tepl_uzg])

            files_6 = files + '\\' + 'Name_file_6.xlsx'
            try:
                Excel_6 = pd.read_excel(files_6)
                end = Excel_6[Excel_6.iloc[:,0] == 'Итого\nза месяц'].index[0]
                df_itog_6 = Excel_6.iloc[end, :]
                podacha_TEG = df_itog_6.iloc[5] 
                potery_TEG = df_itog_6.iloc[4] 
            except:
                Excel_6 = ZeroDivisionError
            
            form_year = np.vstack([date.values, p_in_1st.values, p_out_2st.values, st_sg_1st.values, st_sg_2st.values, p_in_ppa.values, Q_g.values, Ttr.values]).T
            form_year = pd.DataFrame(form_year, columns=['Дата','Р_вх', 'Р_вых','Ст. сж. 1 ст','Ст. сж. 2 ст','Р_ППА','расход_ДКС','TTP'])
            res_year = pd.concat([res_year, form_year],ignore_index=True)
            
            form_month = np.vstack([date_m, Q_g_m, W_a_1_st, obor_1st, p_in_1st_m, t_in_1st_m, st_sg_1st_m, p_out_1st_m, t_out_1st_m, p_avo_1st, t_avo_1st, W_a_2_st, obor_2st, p_in_2st_m, t_in_2st_m, st_sg_2st_m, p_out_2st_m, t_out_2st_m, p_avo_2st, t_avo_2st, t_tepl]).T
            form_month = pd.DataFrame(form_month, columns=['Дата','расход_ДКС', 'Кол-во агр_1','N_ob_1','Р_вх_1', 'T_вх_1', 'Ст. сж. 1 ст', 'Р_вых_1', 'T_вых_1', 'P_вых_аво_1','Т_вых_аво_1','Кол-во агр_2', 'N_ob_2','Р_вх_2', 'T_вх_2', 'Ст. сж. 2 ст','Р_вых_2', 'T_вых_2', 'P_вых_аво_2', 'Т_вых_аво_2', 'T_теплооб'])
            res_month = pd.concat([res_month, form_month],ignore_index=True)
            res_month = res_month.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.dtype == 'str' else x)

            form_month_sep = np.vstack([date_m, p_in_kgs, p_out_kgs, t_in_kgs, t_out_kgs, Q_g_m, W_a_kgs, BMC_kgs, t_out_kgs, t_kgs_tepl, Q_1kgs]).T
            form_month_sep = pd.DataFrame(form_month_sep, columns=['Дата', 'Р_вх_1', 'Р_вых_1', 'T_вх_1', 'T_вых_1', 'расход_ДКС', 'Кол-во агр', 'Объем отс. ВМС', 'T_вх_тепл', 'T_вых_тепл','Q_1сеп'])
            res_month_sep = pd.concat([res_month_sep,form_month_sep],ignore_index=True)
            res_month_sep = res_month_sep.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.dtype == 'str' else x)

            form_month_abs = np.vstack([date_m,  Q_g_m, p_in_abs, t_abs, W_a_abs, Q_1abs, podacha_TEG, potery_TEG, Ttr_m]).T
            form_month_abs = pd.DataFrame(form_month_abs, columns=['Дата', 'расход_ДКС','Р_вх_1', 'T_вх_1', 'Кол-во агр', 'Q_1абс', 'Подача ТЭГ', 'Потери ТЭГ', 'TТР'])
            res_month_abs = pd.concat([res_month_abs,form_month_abs],ignore_index=True)
            res_month_abs = res_month_abs.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.dtype == 'str' else x)

            form_month_potery_1 = np.vstack([date_m, p_in_ppa_m, p_in_kgs, dp_ppa, p_out_kgs, dp_sep, p_out_tepl, dp_tepl, p_in_1st_m, dp_tepl_1st, p_out_1st_m, p_nagn, dp_1st_nagn, p_avo_1st, dp_nagn_avo_1, p_in_2st_m, dp_avo_1_2st, p_out_2st_m, dp_all_ppa_gpa]).T
            form_month_potery_1 = pd.DataFrame(form_month_potery_1, columns=['Дата', 'Р_вх_ппа','Р_вых_ппа', 'dР_ппа','Р_вых_sep','dР_sep', 'Р_вых_tepl', 'dР_tepl', 'Р_вх_1', 'dР_tepl-1st','Р_вых_1', 'Р_нагнет', 'dР_1st_нагн', 'Р_аво', 'dР_нагн_аво1', 'Р_вх_2', 'dР_аво1_2ст','Р_вых_2','dР_все_потери'])
            res_month_potery_1 = pd.concat([res_month_potery_1,form_month_potery_1],ignore_index=True)
            res_month_potery_1 = res_month_potery_1.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.dtype == 'str' else x)    

            form_month_potery_2 = np.vstack([date_m, p_out_2st_m, p_avo_2st, dp_avo_2_tepl, p_in_tepl, p_out_tepl_2, dp_tepl_2, dp_all_gpa_tepl]).T
            form_month_potery_2 = pd.DataFrame(form_month_potery_2, columns=['Дата', 'Р_вых_2', 'P_вых_аво_2', 'dР_avo_tepl', 'Р_вх_tepl','Р_вых_tepl','dР_tepl', 'dР_все_потери'])
            res_month_potery_2 = pd.concat([res_month_potery_2,form_month_potery_2],ignore_index=True)
            res_month_potery_2 = res_month_potery_2.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.dtype == 'str' else x)    

            form_month_potery_3 = np.vstack([date_m, p_out_tepl_2, p_out_abs, dp_tepl_abs, p_out_uzg, dp_abs_uzg, dp_all_tepl_uzg, dp_all]).T
            form_month_potery_3 = pd.DataFrame(form_month_potery_3, columns=['Дата','Р_вых_tepl', 'Р_вых_abs', 'dР_tepl_abs', 'Р_вых_uzg', 'dР_abs_uzg', 'dР_all_tepl_uzg', 'dР_all'])
            res_month_potery_3 = pd.concat([res_month_potery_3,form_month_potery_3],ignore_index=True)
            res_month_potery_3 = res_month_potery_3.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.dtype == 'str' else x)  

        rab_excel = pd.ExcelWriter("Name_file_res.xlsx")
        res_year.to_excel(rab_excel, sheet_name='год', index=False)
        res_month.to_excel(rab_excel, sheet_name='месяц', index=False)
        res_month_sep.to_excel(rab_excel, sheet_name='сепараторы', index=False)
        res_month_abs.to_excel(rab_excel, sheet_name='абсорберы', index=False)
        res_month_potery_1.to_excel(rab_excel, sheet_name='потери от ППА до ГПА', index=False)
        res_month_potery_2.to_excel(rab_excel, sheet_name='потери от ГПА до теплообм.', index=False)
        res_month_potery_3.to_excel(rab_excel, sheet_name='потери от теплообм. до УЗГ', index=False)
        rab_excel.save()

        return res_year, res_month, res_month_sep, res_month_abs
    

    def get_volume_rate(self, res_month):
        Ppr = res_month['Р_вх'] / 4.636
        Tpr = (res_month['T_вх'] + 273.15) / 193.4
        Zpr = 1 - 0.427 * Ppr * Tpr**(-3.688)
        po_vh_1st = res_month['Р_вх'] * 10**6 / (Zpr * 511 * Tpr * 193.4)
        M_1st = (res_month['расход_ДКС'] * 10**6 * 0.682) / (3600 * 24) 
        Q_rate = (M_1st / po_vh_1st) * 60 / res_month['кол-во агр']
        res_month['Q'] = Q_rate
        return res_month
    

    def get_plot_all(self, res_year):
        self.plot_builder.get_p_in(res_year)
        self.plot_builder.get_p_out(res_year)
        self.plot_builder.get_p_in_ppa(res_year)
        self.plot_builder.get_p_in_ppa_with_q_rate(res_year)
        self.plot_builder.get_ttr(res_year)
        names_st = ['Ст. сж. 1 ст','Ст. сж. 2 ст']
        titles = ['КЦ-2', 'КЦ-1']
        for name_st, title in zip(names_st, titles):
            self.plot_builder.get_comp(res_year, name_st, title)


    def get_plot_gdh(self, res_month):
        
        sheet_names = [
            "name"
        ]
        names = [
            "name"
        ]
        for sheet_name, name in sheet_names, names:
            # res_month = 
            data = pd.read_excel('Name_file.xlsx', sheet_name=sheet_name) 
            self.class_gdh.scat_in_plot_gdh(res_month, data, name)


class GetSpch:
    def __init__(self):
        self.plot_builder = PlotBuilder() 

    def parse_gdh_data(self, data):
        #n/nном
        data = data.rename({'Таблица.x.1 Результаты расчета xарактеристики при n/nном=const': 'Параметр', 'Unnamed: 1': 'кпд'}, axis=1)
        end_row_1 = data[data.iloc[:, 0].isna()].index.min()
        end_col_1 = len(data.iloc[:end_row_1, :].dropna(axis=1).columns)
        data1 = data.iloc[1:end_row_1, :end_col_1+1]
        data1.iloc[:, 1] = data1.iloc[:, 1].fillna(method='ffill')
        data1 = data1[data1['Параметр'].isin(['Q, м3/мин', 'e'])]
        # nпол
        end_row_2 = data[data.iloc[:, 0].isna()].index.max()
        end_col_2 = len(data.iloc[end_row_1+2:end_row_2, :].dropna(axis=1).columns)
        data2 = data.iloc[end_row_1+3:end_row_2, :end_col_2+1]
        data2.iloc[:, 1] = data2.iloc[:, 1].fillna(method='ffill')
        data2 = data2[data2['Параметр'].isin(['Q, м3/мин', 'e'])]
        # N/pн
        end_col_3 = len(data.iloc[end_row_2+3:, :].dropna(axis=1, thresh=1).columns)
        data3 = data.iloc[end_row_2+3:, :end_col_3]
        data3.iloc[:, 1] = data3.iloc[:, 1].fillna(method='ffill')
        data3 = data3[data3['Параметр'].isin(['Q, м3/мин', 'e'])]
        row_index = 1
        col_index = 2       
        if row_index < data3.shape[0] and col_index < data3.shape[1]:
            data3 = data3
        else:
            data3 = data3.drop(data3.iloc[:,:2], axis=1)
        return [data1, data2, data3]


    def get_plot_gdh(self, data1, data2, data3, name):
        fig, ax = plt.subplots()
        for kpd in data1['кпд'].unique():
            sup_data = data1[data1['кпд']==kpd]
            x = sup_data.iloc[0, 2:].astype(float).values
            y = sup_data.iloc[1, 2:].astype(float).values
            degree = 5  # Степень полинома
            coefficients = np.polyfit(x, y, degree)
            polynomial = np.poly1d(coefficients)   
            x_smooth = np.linspace(x.min(), x.max(), 300)
            y_smooth = polynomial(x_smooth)       
            ax.plot(x_smooth, y_smooth, color = 'black', linestyle='-')      
            if kpd == 1: 
                ax.plot(x_smooth, y_smooth, linestyle='-', color = 'deepskyblue')
            if kpd == 1.05: 
                ax.plot(x_smooth, y_smooth, linestyle='-', color = 'red')
            ax.text(sup_data.iloc[0, 2], sup_data.iloc[1, 2], round(kpd, 2), ha = 'right', va = 'baseline', fontsize=9)
        for kpd in data2['кпд'].unique():
            sup_data = data2[data2['кпд']==kpd]
            x = sup_data.iloc[0, 2:].astype(float).values
            y = sup_data.iloc[1, 2:].astype(float).values
            degree = 5  # Степень полинома
            coefficients = np.polyfit(x, y, degree)
            polynomial = np.poly1d(coefficients)   
            x_smooth = np.linspace(x.min(), x.max(), 300)
            y_smooth = polynomial(x_smooth)         
            ax.plot(x_smooth, y_smooth, color = 'black', linestyle='-', alpha=0.5)
            ax.text(sup_data.iloc[0, 2:][-1], sup_data.iloc[1, 2:][-1], round(kpd, 2), ha = 'left', va = 'bottom', fontsize=9)
        if data3.empty:
           print(None)
        else: 
            for kpd in data3['кпд'].unique():
                sup_data = data3[data3['кпд']==kpd]
                # x = sup_data.iloc[0, 2:].astype(float).values
                # y = sup_data.iloc[1, 2:].astype(float).values
                # degree = 3  # Степень полинома
                # coefficients = np.polyfit(x, y, degree)
                # polynomial = np.poly1d(coefficients)   
                # x_smooth = np.linspace(x.min(), x.max(), 300)
                # y_smooth = polynomial(x_smooth) 
                ax.plot(sup_data.iloc[0, 2:], sup_data.iloc[1, 2:], color = 'black', linestyle='--', linewidth=1)
                ax.text(sup_data.iloc[0, 2:][-1], sup_data.iloc[1, 2:][-1], round(kpd, 2), ha = 'left', va = 'top', fontsize=9)
        plt.title(f"{name}", fontsize=10)
        ax.set_xlabel('Q, м\u00b3/мин', fontsize=10, loc='right')
        ax.set_ylabel(r'$\epsilon$', fontsize=10, rotation=0, loc='top')
        ax.set_axisbelow(True)
        ax.grid(color='lightgray', linestyle='dashed')
        return fig, ax


    def scat_in_plot_gdh(self, res_month, data, name):
        # res_month['Дата'] = res_month.index
        res_month['color'] = res_month['Дата'].apply(self.plot_builder.set_color_month)
        fig = self.get_plot_gdh(*self.parse_gdh_data(data), name)
        for Q_rate, comp, c, date in zip(res_month['Q'], res_month['comp'], res_month['color'], res_month['Дата']):
            plt.scatter(Q_rate, comp, c=c, label=date.strftime('%B %Y г.'), edgecolors='black')
        plt.legend(labelspacing=0.2, prop={'size':8}, loc='upper right', bbox_to_anchor=(1.26, 1.02))
        return plt

class PlotBuilder:

    def _setup_plot(self, figsize=(9, 3.5)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_axisbelow(True)
        ax.grid(color='lightgray', linestyle='dashed')
        return fig, ax


    def set_color(date):
        color_dict = {
            1: 'royalblue',
            2: 'navy',
            3: 'darkslateblue',
            4: 'steelblue',
            5: 'lightskyblue',
            6: 'darkturquoise',
            7: 'deepskyblue',
            8: 'cornflowerblue',
            9: 'dodgerblue',
            10: 'blue',
            11: 'lightseagreen',
            12: 'paleturquoise'
        }
        month = date.month
        return color_dict[month]
    

    def get_p_in(self, res_year):
        res_year['Дата'] = pd.to_datetime(res_year['Дата'], dayfirst=True)
        date_min = res_year['Дата'].dt.date.min()
        date_max = res_year['Дата'].dt.date.max()
        res_year['color'] = res_year['Дата'].apply(self.set_color)
        #Р_вх
        res_year['month'] = res_year['Дата'].dt.month
        res_year['P_min'] = res_year.groupby('month', as_index=False)['Р_вх'].transform(min) #ищем максимумы и минимумы по месяцам
        res_year['P_max'] = res_year.groupby('month', as_index=False)['Р_вх'].transform(max)
        df_mark = res_year.groupby('month').agg({'Дата': 'mean','P_min':'min', 'P_max':'max'}) #местоположение подписей на графике
        fig, ax = self._setup_plot()
        plt.fill_between(res_year['Дата'], res_year['P_min']-0.03, res_year['P_max']+0.02, step="mid", alpha=0.3, hatch='+', color = 'white', edgecolor='grey') #заливка в выбранной области
        plt.scatter(res_year['Дата'], res_year['Р_вх'], c=res_year['color'],edgecolors='black', s=15) #наносим точки
        plt.plot(res_year['Дата'], res_year['P_min']-0.03, color='darkturquoise', linestyle='--', label='P min, МПа') #линии мин и макс
        plt.plot(res_year['Дата'], res_year['P_max']+0.03, color='b', alpha=0.8, linestyle='--', label='P max, МПа') 
        bbox_min = dict(facecolor='darkturquoise',alpha=0.5) #заливка подписей на графике
        bbox_max = dict(facecolor='royalblue',alpha=0.7)
        df_mark.apply(lambda d: ax.text(d['Дата'],d['P_min']-0.18, round(d['P_min'],2), bbox=bbox_min, va='center', ha = 'center', fontsize=8.5), axis=1) #подписи на графике:местоположение, значения, формат   
        df_mark.apply(lambda d: ax.text(d['Дата'],d['P_max']+0.18, round(d['P_max'],2), bbox=bbox_max, va='center', ha = 'center', fontsize=8.5), axis=1)  
        ax.xaxis.set_major_formatter(formatter) #формат даты
        ax.set_xlim([date_min,date_max]) #пределы по х
        ax.set_ylim(0.5,3)#пределы по у
        ax.set_ylabel('Рвх, МПа', fontsize=12) #название оси у
        plt.legend(fontsize = 10,bbox_to_anchor=(0.65, 1.15), ncol=2) #легенда
        plt.savefig('P_входа.jpg',bbox_inches="tight", dpi=200)


    def get_p_out(self, res_year):
        res_year['Дата'] = pd.to_datetime(res_year['Дата'], dayfirst=True)
        date_min = res_year['Дата'].dt.date.min()
        date_max = res_year['Дата'].dt.date.max()
        res_year['color'] = res_year['Дата'].apply(self.set_color)
        res_year['Р_вых'] = res_year['Р_вых'][res_year['Р_вых'] > 5]
        res_year['P_min_vih'] = res_year.groupby('month', as_index=False)['Р_вых'].transform(min) #ищем максимумы и минимумы по месяцам
        res_year['P_max_vih'] = res_year.groupby('month', as_index=False)['Р_вых'].transform(max)
        df_mark = res_year.groupby('month').agg({'Дата': 'mean','P_min_vih':'min', 'P_max_vih':'max'}) #местоположение подписей на графике
        fig, ax = self._setup_plot()
        plt.fill_between(res_year['Дата'], res_year['P_min_vih']-0.03, res_year['P_max_vih']+0.02, step="mid", alpha=0.3, hatch='+', color = 'white', edgecolor='grey') #заливка в выбранной области
        plt.scatter(res_year['Дата'], res_year['Р_вых'], c=res_year['color'],edgecolors='black', s=15) #наносим точки
        plt.plot(res_year['Дата'], res_year['P_min_vih']-0.03, color='darkturquoise', linestyle='--', label='P min, МПа') #линии мин и макс
        plt.plot(res_year['Дата'], res_year['P_max_vih']+0.03, color='b', alpha=0.8, linestyle='--', label='P max, МПа') 
        bbox_min = dict(facecolor='darkturquoise',alpha=0.5) #заливка подписей на графике
        bbox_max = dict(facecolor='royalblue',alpha=0.7)
        df_mark.apply(lambda d: ax.text(d['Дата'],d['P_min_vih']-0.18, round(d['P_min_vih'],2), bbox=bbox_min, va='center', ha = 'center', fontsize=8.5), axis=1) #подписи на графике:местоположение, значения, формат   
        df_mark.apply(lambda d: ax.text(d['Дата'],d['P_max_vih']+0.18, round(d['P_max_vih'],2), bbox=bbox_max, va='center', ha = 'center', fontsize=8.5), axis=1)  
        ax.xaxis.set_major_formatter(formatter) #формат даты
        ax.set_xlim([date_min,date_max]) #пределы по х
        ax.set_ylim(4.5,7)#пределы по у
        ax.set_ylabel('Рвых, МПа', fontsize=12) #название оси у
        plt.legend(fontsize = 10,bbox_to_anchor=(0.65, 1.15), ncol=2) #легенда
        plt.savefig('P_выхода.jpg',bbox_inches="tight", dpi=200)


    def get_p_in_ppa(self, res_year):
        res_year['Дата'] = pd.to_datetime(res_year['Дата'], dayfirst=True)
        date_min = res_year['Дата'].dt.date.min()
        date_max = res_year['Дата'].dt.date.max()
        res_year['color'] = res_year['Дата'].apply(self.set_color)
        res_year['Р_ППА'] = res_year['Р_ППА'][res_year['Р_ППА'] < 2.5]
        res_year['P_min_ppa'] = res_year.groupby('month', as_index=False)['Р_ППА'].transform(min) #ищем максимумы и минимумы по месяцам
        res_year['P_max_ppa'] = res_year.groupby('month', as_index=False)['Р_ППА'].transform(max)
        df_mark = res_year.groupby('month').agg({'Дата': 'mean','P_min_ppa':'min', 'P_max_ppa':'max'}) #местоположение подписей на графике
        fig, ax = self._setup_plot()
        plt.fill_between(res_year['Дата'], res_year['P_min_ppa']-0.03, res_year['P_max_ppa']+0.02, step="mid", alpha=0.3, hatch='+', color = 'white', edgecolor='grey') #заливка в выбранной области
        plt.scatter(res_year['Дата'], res_year['Р_ППА'], c=res_year['color'],edgecolors='black', s=15) #наносим точки
        ax.plot(res_year['Дата'], res_year['P_min_ppa']-0.03, color='darkturquoise', linestyle='--', label='P min, МПа') #линии мин и макс
        ax.plot(res_year['Дата'], res_year['P_max_ppa']+0.03, color='b', alpha=0.8, linestyle='--', label='P max, МПа') 
        bbox_min = dict(facecolor='darkturquoise',alpha=0.5) #заливка подписей на графике
        bbox_max = dict(facecolor='royalblue',alpha=0.7)
        df_mark.apply(lambda d: ax.text(d['Дата'],d['P_min_ppa']-0.18, round(d['P_min_ppa'],2), bbox=bbox_min, va='center', ha = 'center', fontsize=8.5), axis=1) #подписи на графике:местоположение, значения, формат   
        df_mark.apply(lambda d: ax.text(d['Дата'],d['P_max_ppa']+0.18, round(d['P_max_ppa'],2), bbox=bbox_max, va='center', ha = 'center', fontsize=8.5), axis=1) 
        dp = (res_year['P_max_ppa'][0] - res_year['P_min_ppa'][364]).__round__(2) 
        text = 'ΔP='
        plt.text(res_year.Дата.values[325], 2.75, f'{text}{dp}', fontsize=12, bbox={'boxstyle':'square', 'facecolor': 'darkturquoise', 'alpha':0.3})  
        ax.xaxis.set_major_formatter(formatter) #формат даты
        ax.set_xlim([date_min,date_max]) #пределы по х
        ax.set_ylim(0.5,3)#пределы по у
        ax.set_ylabel('Рвх в ППА, МПа', fontsize=12) #название оси у
        plt.legend(fontsize = 10,bbox_to_anchor=(0.65, 1.15), ncol=2) #легенда
        plt.savefig('P_вх в ппа.jpg',bbox_inches="tight", dpi=200)  


    def get_p_in_ppa_with_q_rate(self, res_year):
        res_year['Дата'] = pd.to_datetime(res_year['Дата'], dayfirst=True)
        date_min = res_year['Дата'].dt.date.min()
        date_max = res_year['Дата'].dt.date.max()
        res_year['color'] = res_year['Дата'].apply(self.set_color)
        res_year['Р_ППА'] = res_year['Р_ППА'][res_year['Р_ППА'] < 2.5]
        res_year['P_min_ppa'] = res_year.groupby('month', as_index=False)['Р_ППА'].transform(min) #ищем максимумы и минимумы по месяцам
        res_year['P_max_ppa'] = res_year.groupby('month', as_index=False)['Р_ППА'].transform(max)
        df_mark = res_year.groupby('month').agg({'Дата': 'mean','P_min_ppa':'min', 'P_max_ppa':'max'}) #местоположение подписей на графике
        fig_1, ax = self._setup_plot()
        plt.fill_between(res_year['Дата'], res_year['P_min_ppa']-0.03, res_year['P_max_ppa']+0.02, step="mid", alpha=0.3, hatch='+', color = 'white', edgecolor='grey') #заливка в выбранной области
        plt.scatter(res_year['Дата'], res_year['Р_ППА'], c=res_year['color'],edgecolors='black', s=15) #наносим точки
        x1 = ax.plot(res_year['Дата'], res_year['P_min_ppa']-0.03, color='darkturquoise', linestyle='--', label='P min, МПа') #линии мин и макс
        x2 = ax.plot(res_year['Дата'], res_year['P_max_ppa']+0.03, color='b', alpha=0.8, linestyle='--', label='P max, МПа') 
        ax2 = ax.twinx() #делаем дополнительную ось
        x3=ax2.plot(res_year['Дата'], res_year['расход_ДКС'], c='dodgerblue',label='Суммарный расход газа')
        ax2.set_ylim(0,80)#пределы по у_2
        ax2.set_ylabel('Q, млн. м\u00b3/сут', fontsize= 12)
        bbox_min = dict(facecolor='darkturquoise',alpha=0.5) #заливка подписей на графике
        bbox_max = dict(facecolor='royalblue',alpha=0.7)
        df_mark.apply(lambda d: ax.text(d['Дата'],d['P_min_ppa']-0.18, round(d['P_min_ppa'],2), bbox=bbox_min, va='center', ha = 'center', fontsize=8.5), axis=1) #подписи на графике:местоположение, значения, формат   
        df_mark.apply(lambda d: ax.text(d['Дата'],d['P_max_ppa']+0.18, round(d['P_max_ppa'],2), bbox=bbox_max, va='center', ha = 'center', fontsize=8.5), axis=1) 
        ax.xaxis.set_major_formatter(formatter) #формат даты
        ax.set_xlim([date_min,date_max]) #пределы по х
        ax.set_ylim(0.5,3)#пределы по у
        ax.set_ylabel('Рвх в ППА, МПа', fontsize=12) #название оси у
        labels = [x1[0],x2[0],x3[0]] 
        leg = [l.get_label() for l in labels]
        ax.legend(labels, leg,bbox_to_anchor=(0.85, 1.15), ncol=3)
        plt.savefig('P_ppa.jpg',bbox_inches="tight", dpi=200)


    def get_ttr(self, res_year):
        res_year['Дата'] = pd.to_datetime(res_year['Дата'], dayfirst=True)
        date_min = res_year['Дата'].dt.date.min()
        date_max = res_year['Дата'].dt.date.max()
        res_year['color'] = res_year['Дата'].apply(self.set_color)
        res_year['TTP'] = res_year['TTP'][res_year['TTP'] < -14]
        res_year['TTP_kr_s'] = res_year['Дата'].apply(lambda d: -14 if 5<=d.month<=9 else None)
        res_year['TTP_kr_w'] = res_year['Дата'].apply(lambda d: -20 if (1<=d.month<=4) | (d.month>=10) else None)
        res_year['TTP_min'] = res_year.groupby('month', as_index=False)['TTP'].transform(min)
        res_year['TTP_max'] = res_year.groupby('month', as_index=False)['TTP'].transform(max)
        df_mark = res_year.groupby('month').agg({'Дата': 'mean','TTP_min':'min', 'TTP_max':'max'})
        fig, ax = self._setup_plot()
        plt.fill_between(res_year['Дата'], res_year['TTP_min']-0.07, res_year['TTP_max']+0.07,step="pre", alpha=0.3,hatch='+',color = 'white',edgecolor='grey')
        plt.plot(res_year['Дата'], res_year['TTP'], c='dodgerblue')
        plt.plot(res_year['Дата'], res_year['TTP_min']-0.07, color='darkturquoise', linestyle='--', label='ТТP min')
        plt.plot(res_year['Дата'], res_year['TTP_max']+0.07, color='b', alpha=0.8, linestyle='--', label='ТТP max')
        plt.plot(res_year['Дата'], res_year['TTP_kr_s'], color='firebrick', label='ТТP прив.')
        plt.plot(res_year['Дата'], res_year['TTP_kr_w'], color='firebrick')
        bbox_min = dict(facecolor='darkturquoise',alpha=0.5) 
        bbox_max = dict(facecolor='royalblue',alpha=0.7) 
        df_mark.apply(lambda d: ax.text(d['Дата'],d['TTP_min']-1.3, round(d['TTP_min'],2),bbox=bbox_min,va='center',ha = 'center', fontsize=8.5),axis=1)     
        df_mark.apply(lambda d: ax.text(d['Дата'],d['TTP_max']+1.3, round(d['TTP_max'],2),bbox=bbox_max,va='center',ha = 'center', fontsize=8.5),axis=1) 
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlim([date_min,date_max]) #пределы по х
        ax.set_ylim(-30,-10)
        ax.set_ylabel('ТТР, \u00b0С', fontsize=12)
        plt.legend(fontsize = 10,bbox_to_anchor=(0.75, 1.15), ncol=3)
        plt.savefig('ТТР.jpg',bbox_inches="tight", dpi=200)

    def get_comp(self, res_year, name, title):
        res_year['Дата'] = pd.to_datetime(res_year['Дата'], dayfirst=True)
        date_min = res_year['Дата'].dt.date.min()
        date_max = res_year['Дата'].dt.date.max()
        res_year['color'] = res_year['Дата'].apply(self.set_color)
        res_year['E_ном'] = 2.20
        res_year['E_min'] = res_year.groupby('month', as_index=False)[name].transform(min) #ищем максимумы и минимумы по месяцам
        res_year['E_max'] = res_year.groupby('month', as_index=False)[name].transform(max)
        df_mark = res_year.groupby('month').agg({'Дата': 'mean','E_min':'min', 'E_max':'max'}) #местоположение подписей на графике
        fig, ax = self._setup_plot()
        plt.fill_between(res_year['Дата'], res_year['E_min']-0.03, res_year['E_max']+0.02, step="mid", alpha=0.3, hatch='+', color = 'white', edgecolor='grey') #заливка в выбранной области
        plt.scatter(res_year['Дата'], res_year[name], c=res_year['color'],edgecolors='black', s=15) #наносим точки
        plt.plot(res_year['Дата'], res_year['E_min']-0.03, color='darkturquoise', linestyle='--', label='e min') #линии мин и макс
        plt.plot(res_year['Дата'], res_year['E_max']+0.03, color='b', alpha=0.8, linestyle='--', label='e max') 
        ax.plot(res_year['Дата'],res_year['E_ном'],color='firebrick',label='Номинальная степень сжатия', zorder=2)
        bbox_min = dict(facecolor='darkturquoise',alpha=0.5) #заливка подписей на графике
        bbox_max = dict(facecolor='royalblue',alpha=0.7)
        df_mark.apply(lambda d: ax.text(d['Дата'],d['E_min']-0.15, round(d['E_min'],2), bbox=bbox_min, va='center', ha = 'center', fontsize=8.5), axis=1) #подписи на графике:местоположение, значения, формат   
        df_mark.apply(lambda d: ax.text(d['Дата'],d['E_max']+0.15, round(d['E_max'],2), bbox=bbox_max, va='center', ha = 'center', fontsize=8.5), axis=1)  
        ax.xaxis.set_major_formatter(formatter) #формат даты
        ax.set_xlim([date_min, date_max]) #пределы по х
        ax.set_ylim(1,3) #пределы по у
        ax.set_ylabel('$\epsilon$', fontsize=12) #название оси у
        plt.legend(fontsize = 10,bbox_to_anchor=(0.80, 1.15), ncol=3) #легенда
        plt.title(title, fontsize = 15,loc = 'left')

        plt.savefig(f'Степень сжатия/{name}.jpg',bbox_inches="tight", dpi=200)

