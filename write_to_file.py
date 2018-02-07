'''
ResultMatrix must be a Nx2 matrix where ResultMatrix[n][0] is the file name
and ResultMatrix[n][1] is it's class.
'''
def WriteResults(ResultMatrix):
	try:
		with open("output.xml", "w") as F:
			F.write("<output>\n")
			for Result in ResultMatrix:
				f.Write("<image src=\"" + Result[0] + "\" class=\"" + Result[1] + "\"/>")
			F.write("</output>")
	except:
		print("Fail")
	print("\nSuccess\n")
