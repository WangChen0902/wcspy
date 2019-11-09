import os
import sys

path = '/data/dataset/program/label_save/'

line_dict = {}

for root,dirs,files in os.walk(path):
    for file in files:
        file_name = file.split('.')[0]
        file_li = file_name.split('_')
        first_name = file_li[0]+'_'+file_li[1]
        index = file_li[2]
        count = len(open(path+file, 'rU').readlines())
        line_dict[first_name] = 0
        # print(file_name, count)

for root,dirs,files in os.walk(path):
    for file in files:
        file_name = file.split('.')[0]
        file_li = file_name.split('_')
        first_name = file_li[0]+'_'+file_li[1]
        index = file_li[2]
        count = len(open(path+file, 'rU').readlines())
        line_dict[first_name] = line_dict[first_name] + count
        print(file_name, count)

out = open('./out_count.txt', 'w')
for l in line_dict:
	out_str = l + ' ' + str(line_dict[l]) + '\n'
	out.write(out_str)