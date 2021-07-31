import psycopg2
import os


path = './classes/'

try:
	connection = psycopg2.connect(user = "",
		                          password = "",
		                          host = "",
		                          port = "5432",
		                          database = "d6o8d3jlfga730")
	#Setting auto commit false
	connection.autocommit = True
	cursor = connection.cursor()
	sql = 'UPDATE pottery SET category = %s WHERE name = %s'
	#cursor.execute(sql, ("xdad", "0403"))
	
		
	# Print PostgreSQL Connection properties
	print ( connection.get_dsn_parameters(),"\n")
	files = os.listdir(path)
	files = [x for x in files if x[-4:] == '.txt']
	for file_index,file in enumerate(files):
		fileName = file.split('.')[0]
		print(file,"\n")
		f = open(path+file, "r")
		for id_name in f:
			#print(id_name) 
			#print(str(fileName), str(id_name))
			#Print PostgreSQL version
			#cursor.execute("SELECT * from pottery where name=%s;",id_name)
			#record = cursor.fetchone()
			# Update
			#cursor.execute("UPDATE pottery SET category = '{0}' WHERE name = '{1}';".format(fileName, id_name))
			#cursor.execute(sql, (str(fileName), str(id_name)))
			id_name = id_name.replace("\n","")
			cursor.execute("UPDATE pottery SET category = '"+fileName+"' WHERE name = '"+id_name+"'")

			#print("UPDATE pottery SET category = '"+str(fileName)+"' WHERE name = '"+str(id_name)+"'")

			#connection.commit()
			#print(cursor.rowcount)
			#break
			#print(record,"\n")
		

except (Exception, psycopg2.Error) as error :
    print ("Error while connecting to PostgreSQL", error)
finally:
    #closing database connection.
        if(connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
