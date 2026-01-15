import pandas as pd 
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import locale
from pathlib import Path
from matplotlib.dates import DateFormatter

mpl.rc('font',family='Times New Roman') #везде TNR
locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8') #русский язык
formatter = DateFormatter('%B \n %Y г.') #формат даты
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


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


    def set_color_month(self, date):
        color_dict = {
            1: 'royalblue',
            2: 'navy',
            3: 'darkslateblue',
            4: 'steelblue',
            5: 'lightskyblue',
            6: 'darkturquoise',
            7: 'deepskyblue',
            8: 'aqua',
            9: 'dodgerblue',
            10: 'blue',
            11: 'lightseagreen',
            12: 'paleturquoise'
        }
        month = date.month
        return color_dict[month]


    def get_p_in(self, res_year):
        res_year['P_in'] = res_year['P_in'].where(res_year['P_in'] < 7)

        res_year['color'] = res_year['Дата'].apply(self.set_color_month)
        res_year['month'] = res_year['Дата'].dt.month
        date_min = res_year['Дата'].dt.date.min()
        date_max = res_year['Дата'].dt.date.max()
        
        res_year['P_min'] = res_year.groupby('month', as_index=False)['P_in'].transform(min) #ищем максимумы и минимумы по месяцам
        res_year['P_max'] = res_year.groupby('month', as_index=False)['P_in'].transform(max)
        df_mark = res_year.groupby('month').agg({'Дата': 'mean','P_min':'min', 'P_max':'max'}) #местоположение подписей на графике

        fig, ax =self._setup_plot() 
        ax.fill_between(res_year['Дата'], res_year['P_min']-0.03, res_year['P_max']+0.03, alpha=0.7, hatch='+', color = 'white', edgecolor='grey') #заливка в выбранной области

        ax.scatter(res_year['Дата'], res_year['P_in'], c=res_year['color'], edgecolors='black', s=15) #наносим точки
        ax.set_ylim(res_year['P_in'].min()-0.5, res_year['P_in'].max()+0.5) #пределы по у

        ax.plot(res_year['Дата'], res_year['P_min']-0.03, color='darkturquoise', linestyle='--', label='P min, МПа') #линии мин и макс
        ax.plot(res_year['Дата'], res_year['P_max']+0.03, color='b', alpha=0.8, linestyle='--', label='P max, МПа') 
        bbox_min = dict(facecolor='darkturquoise', alpha=0.5) #заливка подписей на графике
        bbox_max = dict(facecolor='royalblue', alpha=0.7) 
        df_mark.apply(lambda d: ax.text(d['Дата'], d['P_min']-0.12, round(d['P_min'], 2), bbox=bbox_min, va='center', ha = 'center', fontsize=8.5), axis=1) #подписи на графике:местоположенеи, значения, формат   
        df_mark.apply(lambda d: ax.text(d['Дата'], d['P_max']+0.12, round(d['P_max'], 2), bbox=bbox_max, va='center', ha = 'center', fontsize=8.5), axis=1)   

        ax.xaxis.set_major_formatter(formatter) #формат даты
        ax.set_xlim([date_min, date_max]) #пределы по х
        ax.set_ylabel('Рвх, МПа', fontsize=12) #название оси у
        ax.legend(bbox_to_anchor=(0.70, 1.15), ncol=2)
        return plt


    def get_comp(self, res_year):
        res_year['comp'][1] = res_year['comp'][1].where(res_year['comp'][1] < 5)
        res_year['comp'][2] = res_year['comp'][2].where(res_year['comp'][2] < 5)

        date_min = res_year['Дата'][1].dt.date.min()
        date_max = res_year['Дата'][1].dt.date.max()

        fig, ax = self._setup_plot() 

        if res_year['comp'].loc[1].isna().all():
            ax.scatter(res_year['Дата'][2], res_year['comp'][2], c='deepskyblue', edgecolors='black', s=20, label='1 ступень') #наносим точки     
        else:
            ax.scatter(res_year['Дата'][1], res_year['comp'][1], c='paleturquoise', edgecolors='black', s=20, label='1 ступень') #наносим точки
            ax.scatter(res_year['Дата'][2], res_year['comp'][2], c='deepskyblue', edgecolors='black', s=20, label='2 ступень') #наносим точки          
        
        if res_year['comp_nom'].iloc[2]  == res_year['comp_nom'].iloc[1]:
            ax.plot(res_year['Дата'][2],res_year['comp_nom'][2], color='firebrick', label='Номинальная степень сжатия', zorder=2)
        else: 
            ax.plot(res_year['Дата'][2],res_year['comp_nom'][2], color='firebrick', label='Номинальная степень сжатия 1 ступени', zorder=2)
            ax.plot(res_year['Дата'][1],res_year['comp_nom'][1], color='black', label='Номинальная степень сжатия 2 ступени', zorder=2)

        ax.xaxis.set_major_formatter(formatter) #формат даты
        ax.set_xlim([date_min,date_max]) #пределы по х
        ax.set_ylim(res_year['comp'].min()-0.5, res_year['comp'].max()+0.7)#пределы по у
        ax.set_ylabel('Степень сжатия', fontsize=12) #название оси у
        ax.legend(bbox_to_anchor=(0.75, 1.21), ncol=2)
        return plt


    def get_plot_power(self, res_year):
        res_year['color'] = res_year['Дата'].apply(self.set_color_month)
        res_year['month'] = res_year['Дата'].dt.month
        date_min = res_year['Дата'].dt.date.min()
        date_max = res_year['Дата'].dt.date.max()    

        res_year['P_out'] = res_year['P_out'].where(res_year['P_out'] < 10)
        res_year['power'] = res_year['power'].where(res_year['power'] < 20)
        res_year['power_ном'] = 16
        res_year['power_ном_sum'] = 14.5

        res_year['power_min'] = res_year.groupby('month', as_index=False)['power'].transform(min) #ищем максимумы и минимумы по месяцам
        res_year['power_max'] = res_year.groupby('month', as_index=False)['power'].transform(max)
        df_mark = res_year.groupby('month').agg({'Дата': 'mean','power_min':'min', 'power_max':'max'}) #местоположение подписей на графике

        fig, ax = self._setup_plot() 
        ax.plot(res_year['Дата'], res_year['power_min']-0.03, color='darkturquoise', linestyle='--', label='Минимальная мощность') #линии мин и макс
        ax.plot(res_year['Дата'], res_year['power_max']+0.03, color='b', alpha=0.8, linestyle='--', label='Максимальная мощность') 
        ax.plot(res_year['Дата'], res_year['power_ном'], color='black', label='Номинальная мощность', zorder=2)
        ax.plot(res_year['Дата'], res_year['power_ном_sum'], color='firebrick',label='Располагаемая мощность в летний период', zorder=2)    
        ax.fill_between(res_year['Дата'], res_year['power_min']-0.03, res_year['power_max']+0.02, step="mid", alpha=0.3, hatch='+', color = 'white', edgecolor='grey') #заливка в выбранной области
        ax.scatter(res_year['Дата'], res_year['power'], c=res_year['color'], edgecolors='black', s=15) #наносим точки 
        ax2 = ax.twinx() #делаем дополнительную ось
        ax2.plot(res_year['Дата'], res_year['P_out'], c='dodgerblue', label='Pвых, МПа')
        ax2.set_ylim(0,res_year['P_out'].max()+0.5) #пределы по у_2

        bbox_min = dict(facecolor='darkturquoise',alpha=0.5) #заливка подписей на графике
        bbox_max = dict(facecolor='royalblue',alpha=0.7)
        df_mark.apply(lambda d: ax.text(d['Дата'],d['power_min']-1, round(d['power_min'],2), bbox=bbox_min, va='center', ha = 'center', fontsize=8.5), axis=1) #подписи на графике:местоположение, значения, формат   
        df_mark.apply(lambda d: ax.text(d['Дата'],d['power_max']+1, round(d['power_max'],2), bbox=bbox_max, va='center', ha = 'center', fontsize=8.5), axis=1)  

        ax.xaxis.set_major_formatter(formatter) #формат даты
        ax.set_xlim([date_min,date_max]) #пределы по х
        ax.set_ylim(3, res_year['power_ном'].max() + 4) #пределы по у
        ax.set_ylabel('N, МВт', fontsize=12) #название оси у
        fig.legend(fontsize = 10, bbox_to_anchor=(0.87, 1.05), ncol=3) #легенда
        return plt


    def get_p_out(self, res_year):
        res_year['P_out'] = res_year['P_out'].where(res_year['P_out'] > 3.1)

        res_year['color'] = res_year['Дата'].apply(self.set_color_month)
        res_year['month'] = res_year['Дата'].dt.month
        date_min = res_year['Дата'].dt.date.min()
        date_max = res_year['Дата'].dt.date.max()

        res_year['P_min'] = res_year.groupby('month', as_index=False)['P_out'].transform(min) #ищем максимумы и минимумы по месяцам
        res_year['P_max'] = res_year.groupby('month', as_index=False)['P_out'].transform(max)
        df_mark = res_year.groupby('month').agg({'Дата': 'mean', 'P_min':'min', 'P_max':'max'}) #местоположение подписей на графике

        fig, ax = self._setup_plot() 
        plt.fill_between(res_year['Дата'], res_year['P_min']-0.03, res_year['P_max']+0.03, alpha=0.7,hatch='+',color = 'white',edgecolor='grey') #заливка в выбранной области
        plt.scatter(res_year['Дата'], res_year['P_out'], c=res_year['color'], edgecolors='black', s=15) #наносим точки
        ax.plot(res_year['Дата'], res_year['P_min']-0.03, color='darkturquoise', linestyle='--',label='P min, МПа') #линии мин и макс
        ax.plot(res_year['Дата'], res_year['P_max']+0.03, color='b', alpha=0.8, linestyle='--',label='P max, МПа') 

        bbox_min = dict(facecolor='darkturquoise',alpha=0.5) #заливка подписей на графике
        bbox_max = dict(facecolor='royalblue',alpha=0.7) 
        df_mark.apply(lambda d: ax.text(d['Дата'],d['P_min']-0.15, round(d['P_min'],2),bbox=bbox_min,va='center',ha = 'center', fontsize=8.5),axis=1) #подписи на графике:местоположенеи, значения, формат   
        df_mark.apply(lambda d: ax.text(d['Дата'],d['P_max']+0.15, round(d['P_max'],2),bbox=bbox_max,va='center',ha = 'center', fontsize=8.5),axis=1)   
        
        ax.xaxis.set_major_formatter(formatter) #формат даты
        ax.set_xlim([date_min,date_max]) #пределы по х
        ax.set_ylim(res_year['P_out'].min()-0.5,res_year['P_out'].max()+0.5)#пределы по у
        ax.set_ylabel('Рвых, МПа', fontsize=12) #название оси у
        ax.legend(bbox_to_anchor=(0.70, 1.15), ncol=2)
        return plt


    def get_power_1v(self, res_year):
        res_year['color'] = res_year['Дата'].apply(self.set_color_month)
        res_year['month'] = res_year['Дата'].dt.month
        date_min = res_year['Дата'].dt.date.min()
        date_max = res_year['Дата'].dt.date.max()    

        res_year['P_out'] = res_year['P_out'].where(res_year['P_out'] < 10)
        res_year['power'] = res_year['power'].where(res_year['power'] < 20)
        res_year['power_ном'] = 16
        res_year['power_ном_sum'] = 14.5

        res_year['power_min'] = res_year.groupby('month', as_index=False)['power'].transform(min) #ищем максимумы и минимумы по месяцам
        res_year['power_max'] = res_year.groupby('month', as_index=False)['power'].transform(max)
        df_mark = res_year.groupby('month').agg({'Дата': 'mean','power_min':'min', 'power_max':'max'}) #местоположение подписей на графике

        fig, ax = self._setup_plot() 
        ax.plot(res_year['Дата'], res_year['power_min']-0.1, color='darkturquoise', linestyle='--', label='Минимальная мощность') #линии мин и макс
        ax.plot(res_year['Дата'], res_year['power_max']+0.1, color='b', alpha=0.8, linestyle='--', label='Максимальная мощность') 
        ax.plot(res_year['Дата'], res_year['power_ном'], color='black', label='Номинальная мощность', zorder=2)
        ax.plot(res_year['Дата'], res_year['power_ном_sum'], color='firebrick',label='Располагаемая мощность в летний период', zorder=2)    
        ax.fill_between(res_year['Дата'], res_year['power_min']-0.03, res_year['power_max']+0.02, step="mid", alpha=0.3, hatch='+', color = 'white', edgecolor='grey') #заливка в выбранной области
        ax.scatter(res_year['Дата'], res_year['power'], c=res_year['color'], edgecolors='black', s=15) #наносим точки 
        ax2 = ax.twinx() #делаем дополнительную ось
        ax2.plot(res_year['Дата'], res_year['расход_ДКС'], c='dodgerblue', label='Q, млн. м\u00b3/сут')
        ax2.set_ylim(0,res_year['расход_ДКС'].max()+2) #пределы по у_2

        bbox_min = dict(facecolor='darkturquoise',alpha=0.5) #заливка подписей на графике
        bbox_max = dict(facecolor='royalblue',alpha=0.7)
        df_mark.apply(lambda d: ax.text(d['Дата'],d['power_min']-1, round(d['power_min'],2), bbox=bbox_min, va='center', ha = 'center', fontsize=8.5), axis=1) #подписи на графике:местоположение, значения, формат   
        df_mark.apply(lambda d: ax.text(d['Дата'],d['power_max']+1, round(d['power_max'],2), bbox=bbox_max, va='center', ha = 'center', fontsize=8.5), axis=1)  
        
        ax.xaxis.set_major_formatter(formatter) #формат даты
        ax.set_xlim([date_min,date_max]) #пределы по х
        ax.set_ylim(3, res_year['power_ном'].max() + 4) #пределы по у
        ax.set_ylabel('N, МВт', fontsize=12) #название оси у
        fig.legend(fontsize = 10, bbox_to_anchor=(0.87, 1.05), ncol=3) #легенда
        return plt
    

class YamburgFact:
    def __init__(self):
        self.plot_builder = PlotBuilder() 
        self.class_gdh = GetSpch()
        self.sheet_names_dks = ['ДКС-1', 'ДКС-2', 'ДКС-3','ДКС-4','ДКС-5','ДКС-6','ДКС-7','ДКС-1В','ДКС-9']
    

    @classmethod
    def get_middle_digit(cls, number):
        """Возвращает среднюю цифру трехзначного числа"""
        if isinstance(number, str) and len(number) == 3:
            return int(number[1])
        return None    
    

    @classmethod
    def remove_star(cls, value):
        """Убирает * у значений"""
        if pd.isna(value):
            return value    
        value_str = str(value)
        value_cleaned = value_str.replace('*', '')
        try:
            return int(value_cleaned)
        except ValueError:
            try:
                return float(value_cleaned)  
            except ValueError:
                return value_cleaned


    def share_df(self, df_new):
        last_index = df_new.index[-1]  # Индекс последней строки
        df_new = df_new.drop(last_index)
        # df_new = df_new.dropna(how='all')
        df_new.columns = [['№ДКС', '№ агр.', 'Состояние', 'Pвх', 'Pвых', 'Tвх', 'Tвых', 'Девиа-', 'E', 'NaN', 'NaN', 'частота', 'T за твд', 'N,МВт', 'Q','Дата']]
        df_new = df_new.drop(['Девиа-','NaN', 'NaN','T за твд'], axis=1)
        df_new['№ агр.'] = df_new['№ агр.'].astype(str)
        df_new['name'] = df_new['№ агр.'].apply(lambda numbers:[self.get_middle_digit(num) for num in numbers])  
        columns_to_apply = ['Pвх', 'Pвых', 'Tвх', 'Tвых', 'E', 'частота', 'N,МВт', 'Q']
        df_new[columns_to_apply] = df_new[columns_to_apply].apply(lambda col: col.apply(self.remove_star))
        return df_new

     
    def work_df(self, df):
        """Забор данных из таблицы"""
        idx_1 = (df.name == 1).values
        idx_2 = (df.name == 2).values
        df_1 = df.loc[idx_1]
        df_2 = df.loc[idx_2]
        
        q_rate_2 = df['Q'].iloc[1] * 24 / 1000
        q_rate_1 = df['Q'].iloc[2] * 24 / 1000
        date = df['Дата'].iloc[0]

        def data(df):
            p_in = df['Pвх'].where(df['Pвх'] < 5).mean(skipna=True) + 0.1
            p_out = df['Pвых'].where(df['Pвых'] < 9).mean(skipna=True) + 0.1
            t_in = df['Tвх'].mean(skipna=True)
            t_out = df['Tвых'].mean(skipna=True) 
            comp = df['E'].where(df['E'] > 1.001).mean(skipna=True) 
            freq = df['частота'].mean(skipna=True)  
            power = df['N,МВт'].mean(skipna=True)    
            cnt = (df['Pвх'] != 0).sum()
            cnt = df['Pвх'].count(axis=0)
            cnt = cnt.where(cnt != 0, np.nan)
            df_st = pd.concat([cnt, freq, p_in, p_out, comp, t_in, t_out, power])
            return df_st
        
        df_2_st = data(df_1)
        df_2_st_with_q = pd.concat([date, q_rate_2, df_2_st],axis=0)

        if df_2.empty:
            df_1_st_with_q = pd.Series(date)
        else:
            df_1_st = data(df_2)
            df_1_st_with_q = pd.concat([date, q_rate_1, df_1_st],axis=0)
        df_ob = pd.concat([df_2_st_with_q, df_1_st_with_q], keys=[2,1], axis=1)
        return df_ob


    def get_volume_rate(self, res_year):
        Ppr = res_year['P_in'] / 4.636
        Tpr = (res_year['T_in'] + 273.15) / 193.4
        Zpr = 1 - 0.427 * Ppr * Tpr**(-3.688)
        po_vh_1st = res_year['P_in'] * 10**6 / (Zpr * 511 * Tpr * 193.4)
        M_1st = (res_year['расход_ДКС'] * 10**6 * 0.682) / (3600 * 24) 
        Q_rate = (M_1st / po_vh_1st) * 60 / res_year['кол-во агр']
        res_year['Q'] = Q_rate
        return res_year
    

    def get_res(self, path_data):
        os.makedirs('names', exist_ok=True)
        folders = Path(path_data)
        file_list = list(folders.glob('*.xlsx'))
        dict_df = {}
        dict_total = {}

        for file in file_list:
            keys = file.name
            list_df = []
            list_total = []
            Excel = pd.read_excel(file, sheet_name=None)
            for key in Excel:
                Excel[key].replace({0:np.nan}, inplace=True)
                Excel[key].replace({'...':np.nan}, inplace=True)
                date = pd.to_datetime(key, dayfirst=True, format='%d.%m.%Y')
                Excel[key].columns = Excel[key].iloc[1,:]
                Excel_rab = Excel[key].iloc[5:,:15].reset_index(drop=True)
                end_row = Excel_rab[Excel_rab.iloc[:, 0].isna()].index
                filtered_df = Excel_rab[Excel_rab['№'].str.contains(r'ДКС', na=False, regex=True)]
                ind_df = filtered_df.index.append(end_row[-1:])
                for i in range(len(ind_df)-1):
                    df_new = Excel_rab[ind_df[i]:ind_df[i+1]]
                    df_new['Дата'] = date
                    df_share = self.share_df(df_new)
                    df_work = self.work_df(df_share).T
                    df_work.columns = ['Дата', 'расход_ДКС', 'кол-во агр','freq', 'P_in', 'P_out','comp', 'T_in', 'T_out', 'power']
                    list_df.append(df_work)     

            dict_df[keys] = list_df
            len_list = len(dict_df[keys])
            for j in range(9):
                df_total = pd.concat([dict_df[keys][i:i+9][j] for i in range(0, len_list, 9)])
                list_total.append(df_total)
            dict_total[keys] = list_total

        #лист с датафреймами каждой ДКС
        list_res = []
        for i in range(9):
            list_of_dfs = [dict_total[keys][i] for keys in dict_total]
            result_df = pd.concat(list_of_dfs)
            list_res.append(result_df)
        list_res[7].loc[list_res[7].index == 1, 'расход_ДКС'] = list_res[7].loc[list_res[7].index == 2, 'расход_ДКС'].values
        return list_res
    

    def get_month_data(self, list_res, path_mer):
        rab_excel = pd.ExcelWriter("names/names_all.xlsx")

        dict_dks_all_month = {}

        for df, sheet_name in zip(list_res, self.sheet_names_dks):
            list_dks_all_month = []
            
            for i in [1,2]:
                df_mini = df.loc[i]
                df_mini['Период'] = pd.PeriodIndex(df_mini['Дата'], freq="M")
                df_month = df_mini.groupby('Период')[['Дата']].mean()
                df_month['расход_ДКС'] = df_mini.groupby('Период')['расход_ДКС'].mean().round(2)
                df_month['кол-во агр'] = df_mini.groupby('Период')['кол-во агр'].mean().round(0)
                df_month['freq'] = df_mini.groupby('Период')['freq'].mean().round(0)
                df_month[['P_in', 'P_out', 'comp',  'T_in', 'T_out', 'power']] = df_mini.groupby('Период')[['P_in', 'P_out', 'comp', 'T_in', 'T_out','power']].mean().round(2)
                list_dks_all_month.append(df_month)

            dict_dks_all_month[sheet_name] = list_dks_all_month
            if (sheet_name != 'ДКС-1В') & (sheet_name != 'ДКС-9') & (sheet_name != 'ДКС-4'):
                dict_dks_all_month[sheet_name][0]['расход_ДКС'] = dict_dks_all_month[sheet_name][1]['расход_ДКС']

            df_yamb_res = pd.concat(dict_dks_all_month[sheet_name], axis=1, keys=[1,2])
            df_yamb_res = df_yamb_res.stack(level=0)[['кол-во агр','freq', 'P_in', 'P_out', 'comp','T_in', 'T_out', 'расход_ДКС','power']]
            new_index_level  = df_yamb_res.index.levels[0].strftime('%B %Y г.')
            df_yamb_res.index = df_yamb_res.index.set_levels(new_index_level, level=0)    
            df_yamb_res.to_excel(rab_excel, sheet_name = sheet_name)

        folders_mer = Path(path_mer)
        file_list_mer = list(folders_mer.glob('*.xlsx'))
        lst_q_rate_dks_4 = []
        for file_mer in file_list_mer:
            Excel = pd.read_excel(file_mer)
            filtered_df = Excel[Excel.iloc[:,0].str.contains(r'УКПГ-4', na=False, regex=True)].index
            q_rate = Excel.iloc[filtered_df, 9]
            lst_q_rate_dks_4.append(q_rate)
        q_rate_dks_4 = pd.concat(lst_q_rate_dks_4)
        q_rate_dks_4 = q_rate_dks_4.to_frame(name='расход_ДКС').set_index(df_month.index)
        dict_dks_all_month['ДКС-4'][0]['расход_ДКС'] = q_rate_dks_4
        return rab_excel.save() 
    

    def get_plot_all(self, list_res):

        for df in list_res:
            df['comp_nom'] = 3
        list_res[3]['comp_nom'].loc[1] = 2.2 
        list_res[6]['comp_nom'].loc[1] = 2.2    
        list_res[7]['comp_nom'].loc[1] = 2.2 
        list_res[7]['comp_nom'].loc[2] = 2.2    
        list_res[8]['comp_nom'] = 2.2  
        list_res[8].loc[1] = list_res[8].loc[1].fillna(np.nan)

        sheet_names = ['1 ступень', '2 ступень']
        for i, sheet_name_dks in zip(range(9), self.sheet_names_dks):
            self.plot_builder.get_comp(list_res[i]).savefig(f'names/Степень сжатия {sheet_name_dks}.jpg',bbox_inches="tight", dpi=200)
            for j, sheet_name in zip([1,2], sheet_names):
                if not (i == 8 and j == 1): 
                    self.plot_builder.get_p_in(list_res[i].loc[j]).savefig(f'names/P_входа {sheet_name} {sheet_name_dks}.jpg',bbox_inches="tight", dpi=200)
                    self.plot_builder.get_plot_power(list_res[i].loc[j]).savefig(f'names/Мощность {sheet_name} {sheet_name_dks}.jpg',bbox_inches="tight", dpi=200)
                else:
                    pass
                
        for i, sheet_name in zip([1,2], sheet_names):
            self.plot_builder.get_p_out(list_res[7].loc[i]).savefig(f'names/P_выхода {sheet_name} ДКС-1В.jpg',bbox_inches="tight", dpi=200)
            self.plot_builder.get_power_1v(list_res[7].loc[i]).savefig(f'names/Мощность {sheet_name} ДКС-1В.jpg',bbox_inches="tight", dpi=200)

    def get_plot_gdh(self, dict_dks_all_month):
        doubled_sheet_name = [key for key in self.sheet_names_dks for _ in range(2)]

        sheet_names_gdh = ["names"]
        
        names_gdh = ["names"]

        for sheet_name, sheet_name_gdh, name_gdh, j  in zip(doubled_sheet_name, sheet_names_gdh, names_gdh, range(len(doubled_sheet_name))):
            i = j % 2  
            dict_dks_all_month[sheet_name][i] = self.get_volume_rate(dict_dks_all_month[sheet_name][i])    
            data = pd.read_excel('files.xlsx', sheet_name=sheet_name_gdh)  
            self.class_gdh.scat_in_plot_gdh(dict_dks_all_month[sheet_name][i], data, name_gdh).savefig(f'names/ГДХ {sheet_name} {i+1} ступень.jpg',bbox_inches="tight", dpi=200) 
