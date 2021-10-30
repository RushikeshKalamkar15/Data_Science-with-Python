# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 15:58:59 2021
@author: Rushikesh
"""
#-------------------------Reading & Writing data in Files----------------------

import pandas

# Reading CSV Files with Pandas:
df = pandas.read_csv('E:/Read&Write/User_Data.csv')
print(df)

# Writing CSV Files with Pandas:
df.to_csv('E:/Read&Write/User_Data.csv')

# Reading Excel Files with Pandas
df1 = pandas.read_excel('E:/Read&Write/User_Data.xlsx')

df1 = pandas.read_excel('User_Data.xlsx')
print(df1)

# Writing Excel Files with Pandas 
df1.to_excel('User_Data.xlsx')
df2 = pandas.DataFrame(df1)
print (df2)
