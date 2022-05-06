import pandas as pd
import numpy as np
from numpy import log as ln



def bookfeat(dataframe_book):
    '''
        Funcao com o objetivo de criar os features engineering previamente pensados a partir do dataframe
         com as informacoes de book da acao para uso no modelo preditivo de volatilidade
    '''
    
    dataframe_book['wap'] = (dataframe_book['bid_price1']*dataframe_book['ask_size1']+dataframe_book['ask_price1']*dataframe_book['bid_size1'])/(dataframe_book['bid_size1']+dataframe_book['ask_size1'])
    dataframe_book['wap2'] = (dataframe_book['bid_price2']*dataframe_book['ask_size2']+dataframe_book['ask_price2']*dataframe_book['bid_size2'])/(dataframe_book['bid_size2']+dataframe_book['ask_size2'])
    dataframe_book['tend_vol']= np.where(dataframe_book['wap2']>dataframe_book['wap'],1,0)
    lista_med=[]
    for i in dataframe_book['time_id'].unique():
        df1 = dataframe_book[dataframe_book['time_id']==i]
        for wap in df1['wap']:
            lista_med.append(int(np.where(wap > df1['wap'].mean(),1,0)))
    dataframe_book['acima/abaixo media']=lista_med
    dataframe_book['log_return_ask'] = ln((dataframe_book['ask_price1']-dataframe_book['ask_price2'])/dataframe_book['ask_price2'] + 1)
    dataframe_book['log_return_bid'] = ln((dataframe_book['bid_price1']-dataframe_book['bid_price2'])/dataframe_book['bid_price2'] + 1)
    dataframe_book['log_ask_div_bid'] = ln(dataframe_book['ask_price1']/dataframe_book['bid_price1'])
    dataframe_book['log_vol_ask_div_bid'] = ln(dataframe_book['ask_size1']/dataframe_book['bid_size1'])
    dataframe_book['log_ask_div_ask'] = ln(dataframe_book['ask_price2']/dataframe_book['bid_price1'])
    dataframe_book['log_bid_div_bid'] = ln(dataframe_book['bid_price2']/dataframe_book['bid_price1'])
    dataframe_book['log_vol_ask_div_ask'] = ln(dataframe_book['ask_size2']/dataframe_book['ask_size1'])
    dataframe_book['log_vol_bid_div_bid'] = ln(dataframe_book['bid_size2']/dataframe_book['bid_size1'])
    return dataframe_book
def tradefeat(dataframe_trade):
    '''
        Funcao com o objetivo de criar os features engineering previamente pensados a partir do dataframe
         com as informacoes de trade da acao para uso no modelo preditivo de volatilidade
    '''
    
    dataframe_trade['size/order_count']=dataframe_trade['size']/dataframe_trade['order_count']
    lista_dif =[dataframe_trade['seconds_in_bucket'][0]]
    ponto = dataframe_trade['seconds_in_bucket'][0]
    for i in dataframe_trade['seconds_in_bucket'][1:]:
        x = i-ponto
        lista_dif.append(x)
        ponto=i
    dataframe_trade['intervalo_bucket'] = lista_dif
    dataframe_trade['price*size/order_count'] = dataframe_trade['price']*dataframe_trade['size']/dataframe_trade['order_count']
    dataframe_trade['price*size/order_count*intervalo'] = dataframe_trade['price']*dataframe_trade['size']/(dataframe_trade['order_count']*dataframe_trade['intervalo_bucket'])
    return dataframe_trade 

def max_min(array):
    '''
        Funcao basica para agregacao pegando o maximo e o minimo de uma array
    '''
    return array.max()-array.min()


def aggfunc(dataframe_book,dataframe_trade):
    '''
        Funcao para agregar as informacoes obtidas nos features engineering, com embassamento em metricas estatisticas 
    '''
    df_book_group_by = dataframe_book.groupby('time_id').agg({    
    'wap':['std','mean',max_min],
    'wap2':['std','mean',max_min],
    'seconds_in_bucket': ['std', 'mean',max_min],
    'log_return_ask':['std', 'mean',max_min],
    'log_return_bid':['std', 'mean',max_min],
    'log_ask_div_bid':['std', 'mean',max_min],
    'log_vol_ask_div_bid':['std', 'mean',max_min],
    'log_ask_div_ask':['std', 'mean',max_min],
    'log_bid_div_bid':['std', 'mean',max_min],
    'log_vol_ask_div_ask':['std', 'mean',max_min],
    'log_vol_bid_div_bid':['std', 'mean',max_min],
    'acima/abaixo media':'sum',
    'tend_vol':'sum'})

    df_trade_groub_by = dataframe_trade.groupby('time_id').agg({    
    'intervalo_bucket': ['std', 'mean',max_min],
    'price*size/order_count*intervalo':['std', 'mean',max_min],
    'size/order_count':['std', 'mean']}) 

    df_final = df_book_group_by.merge(df_trade_groub_by.reset_index(), on='time_id')
    df_final.columns = [ '_'.join(x) for x in df_final.columns]
    df_final = df_final.rename({'time_id_':'time_id'}, axis=1)
    return df_final