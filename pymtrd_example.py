# -*- coding: utf-8 -*-
import pymtrd_process as pp

if __name__ == '__main__':
    config_path = r'config.csv'
    num_threads = 10
    bool_draw = 1
    # select the correct function to process, only the pp.process_parallel need the num of threads
    pp.process_parallel(config_path, num_threads, bool_draw)
    # pp.process(config_path, bool_draw)
    # pp.process_draw(config_path)
    print('done!!!')
