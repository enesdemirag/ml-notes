# Burada datayi pandas'a uygun bir sekilde duzenliyorum ve dataseti .csv olarak kaydediyorum. 

with open("mammographic_masses.data", "r") as file:
    data = file.read() # Tum datayi okuyorum

data = data.replace("?", "") # ? yerine Nan

dataset_train = open("mammographic_masses_train.csv", "w")
dataset_test = open("mammographic_masses_test.csv", "w")

# Labellari en basa ekliyorum
dataset_train.write('"BI-RADS","Age","Shape","Margin","Density","Severity"\n')
dataset_test.write('"BI-RADS","Age","Shape","Margin","Density","Severity"\n')

# Testing ve Training icin 80-20 oraninda 2 dataset olusturuyorum.
train_data = data[:int(len(data) * 0.8)+2]
test_data = data[int(len(data) * 0.8)+3:]

dataset_train.write(train_data)
dataset_test.write(test_data)

dataset_train.close()
dataset_test.close()