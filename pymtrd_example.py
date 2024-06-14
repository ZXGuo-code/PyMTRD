# -*- coding: utf-8 -*-
import pandas as pd
import pymtrd_process as pp

if __name__ == '__main__':
    
    df = pd.read_csv('input_new.csv', header=None)
    pp.process(df)
    # df = pd.read_csv('input_draw.csv', header=None)
    # pp.process_draw(df)
    print('done!!!')
