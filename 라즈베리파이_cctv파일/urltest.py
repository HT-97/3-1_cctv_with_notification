from urllib.request import urlopen
import time

c = False
while c == False:
    try:
        res = urlopen('http://google.com')
        print(res.status)
        c = True
    except:
        print("not connected")
        
    time.sleep(2)
    
print("done")