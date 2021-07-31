import os
import csv

origin_path = "./AllModels"
target_path = "./pottery_dataset"


categories = {'Alabastron':1,
'Amphora':2,
'Hydria':12,
'Kalathos':13,
'Krater':16,
'Kylix':18,
'Lekythos':22,
'Native-American':47,
'Pelike':29,
'Picher-Shaped':48,
'Psykter':32
}



for cat in categories:
	command = " ".join(["mkdir",target_path+"/"+cat])
	os.system(command)
	f_train_command = " ".join(["mkdir",target_path+"/"+cat+"/train"])
	os.system(f_train_command)
	f_test_command = " ".join(["mkdir",target_path+"/"+cat+"/test"])
	os.system(f_test_command)




reader = csv.DictReader(open("train01.csv"))



#print(cat)


def get_cat(actual_id):
	for key, value in categories.items():
		if value == int(actual_id):
			return key


for row in reader:
	#name = row['mobject_label']
	#name = row['mobject_label'].replace("_", "\ ")
	name = row['mobject_label'].replace(" ", "\ ")

	cat = get_cat(row['class_id'])
	command = " ".join(["cp",origin_path+"/"+name+".obj",target_path+"/"+cat+"/"+"train"+"/"+name+".obj"])
	print(command)
	os.system(command)
#print(name,row['class_id'])
#print(row['name'],row['category'],row['split'])
#if(row['category']!='trash'):


