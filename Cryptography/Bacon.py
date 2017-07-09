A = "asdfghjklqwertyuiopzxcvbnm"
B = A.upper()
X = "bacoN is one of aMerICa'S sWEethEartS. it's A dARlinG, SuCCulEnt fOoD tHAt PaIRs FlawLE"
W = ""
for c in X:
    if c in A:
        W+='A'
    elif c in B:
        W+='B'
print(W)
D = {"AAAAA":"a",  "AABBA":"g",  "ABBAA":"n",  "BAABA":"t", 
"AAAAB":"b",  "AABBB":"h",  "ABBAB":"o",  "BAABB":"u-v",
"AAABA":"c",  "ABAAA":"i-j",  "ABBBA":"p",  "BABAA":"w", 
"AAABB":"d",  "ABAAB":"k", "ABBBB":"q",  "BABAB":"x", 
"AABAA":"e",  "ABABA":"l",  "BAAAA":"r",  "BABBA":"y",
"AABAB":"f",  "ABABB":"m",  "BAAAB":"s",  "BABBB":"z" }
Y = ""
for i in range(0,100,5):
	if i+5 <= len(W):
		Y +=D[W[i:i+5]]
print(Y)