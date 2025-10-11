#los 30 pasientes que el clasificador detecto que tenian la enfermedad y y realmente la tenian
TP = 30
#los 20 pasientes que el clasificador clasifico con la enfermedad pero realmente NO la tenian
FP = 20
#los 140 pasientes a los cuales el clasificador clasifico sin la enfermedad y realmente NO la tenian
TN = 140
#los pacientes a los cuales el clasificador no detecto la enfermedad y realmente si la tenian 
FN = 10

       #metricas
#exactitud
accurazy = (TP + TN)/(TP + TN + FP + FN)
print(f"exactitud = {accurazy}")

presicion = TP / (TP + FP)
print(f"presicion = {presicion}")

recall = TP / (TP + FN)
print(f"exahustividad = {recall}")

