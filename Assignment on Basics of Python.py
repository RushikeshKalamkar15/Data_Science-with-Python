# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 10:38:10 2021

@author: Rushikesh
"""
#Assignment on Basic’s of Python :

#Create a list:
l = [1,5,0,7,1,9,9,8]
print(l)
    
#reverse the order of the list.
l = [1,5,0,7,1,9,9,8]
print("l before reverse:",l)
l.reverse()
print("l after reverse:",l)

#append a item in it.
l = [1,5,0,7,1,9,9,8]
print(l)
l.append(50)
print(l)

#extend the two lists into one.
list_1 = [1,3,5]
print(list_1)
list_2 = [2,4,6]
print(list_2)
list_3 = []
print(list_3)
list_3.extend(list_1+list_2)
print(list_3)

#insert a item in list.
home = ['grandmother','father','mother','sister']
print(home)
home.append('brother')
print(home)

#sort the values in ‘descending order’
l = [1,5,0,7,1,9,9,8]
print(l)
l.sort(reverse=True)
print(l)

#remove the last item in the list.
list = ['vishu','isha','siya','hina','alia','leena']
print(list)
del list[5]
print(list)


#Create a string:
str = ('rushikesh')
print(str)

#split the string by (“  “)
sentence = "I am a good person"
words = sentence.split()
print(words)


#replace the values in it.
str1 = ('h,o,m,e')
print(str1)
str1.replace('h', 'l')



#Create a dictionary with keys and values:
subject = {'science': '01', 'math': '02', 'english':'03', 'marathi': '04', 'hindi': '05'}
print(subject)
    
#sort the value’s by using keys or values.
subject = {'science': '01', 'math': '02', 'english':'03', 'marathi': '04', 'hindi': '05'}
print(subject)
sorted(subject)

#merge two dictionaries
sales1 = {'India':100, 'China':500, 'USA':200, 'UK':400, 'Japan':700}
sales2 = {'Argentia':200, 'Canada':300, 'Kenya':100, 'Germany':800, 'France':500}
print(sales1)
print(sales2)
sales = sales1.copy()
sales.update(sales2)
print(sales)

#delete a key in the dictionary.
subject = {'science': '01', 'math': '02', 'english':'03', 'marathi': '04', 'hindi': '05'}
print(subject)
del subject['science']
print(subject)

#access a value by using key of the dictionary.
subject = {'science': '01', 'math': '02', 'english':'03', 'marathi': '04', 'hindi': '05'}
print(subject)
print(len("keys"))


#Create a tuple’s with both string an integers:
str1 = (1998)
str2 = (2000) 
t1 = tuple('1998,2000') 
print('t1=',t1) 
    
#make two tuples as one(i.e add).
t1=(11,12,13,14,15)
t2=(16,17,18,19,20)
print(t1+t2)

#delete a item in tuple.
t1=(11,12,13,14,15)
print(t1)
#tuples  are immutable so there is no deletation in t1


#Convert a tuple into dictionary.
lt=[('Vishal',23),('Vikas',22),('Vilas',21)]
dictionary= dict(lt)
print(dictionary)


#Write functions by using If/ If else / if elif else condition’s 
#If statements
a = int(input("Enter a? "));
b = int(input("Enter b? "));
c = int(input("Enter c? "));
if a>b and a>c:
    print("a is largest");

if b>a and b>c:
    print("b is largest");

if c>a and c>b:
    print("c is largest");
    
#If else statements
age = int(input("Enter your age? "))
if age>=18:
    print("You are eligible for doing job !!");
else:
    print("Sorry! you have to wait !!");
    
#If elif else Statements
number = int(input("Enter the number?"))
if number==15:
    print("number is equals to 15")
elif number==30:
    print("number is equal to 30");
elif number==45:
    print("number is equal to 45");
else:
    print("number is not equal to 15, 30 or 45");



#Write functions by using different loops I.e for loop,while loop and nested loop.
#For Loop
i=20
for i in range(0,15):
    print(i,end =',')

#While Loop
i=1;
while i<=10:
    print(i);
    i=i+1;
    
#Nested Loop
n = int(input("Enter the number of rows you want to print?"))
i,j=0,0
for i in range(0,n):
    print()
    for j in range(0,i+1):
        print("*",end="")
        

#Check whether an item or value is present in the list by using “for loop with if condition”.
mylist = [1,2,3,4,5]
Number=5
for i in mylist:
    if(i==Number):
        print("Number is Found in mylist")


#Import a data into python by using pandas module.
import pandas as pd

#csv format data
df = pd.read_csv('E:/HW_DataVisualization/file1.csv')
print(df)

#excel format data
df = pd.read_excel('E:/HW_DataVisualization/Book1.xlsx')
print(df)


#create 3 lists and make a dataframe with it and perform all 4 moments of business decisions.
import pandas as pd

d = {'Name':['Vishal','Raj','Mayur','Rajat','Piyush'],
'Marks':[98,70,65,78,91],
'Subjects':['Math','Math','Math','Math','Math']}
d
df = pd.DataFrame(d)
print(df)


