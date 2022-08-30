import pickle
import numpy as np
import pandas as pd

data = pickle.load(open('test_pred.pkl', 'rb'))

ans = []

for i in data:
    m = np.argmax(i)
    if m > 0 and abs(i[m-1] - i[m]) < .1 and not (m < 4 and abs(i[m+1] - i[m]) < .1):
        print('-')
        ans.append(m - .5 + 1)
    elif m < 4 and abs(i[m+1] - i[m]) < .1 and not (m > 0 and abs(i[m-1] - i[m]) < .1):
        print('-')
        ans.append(m + .5 + 1)
    else:
        ans.append(m + 1)

# data = np.array(data).argmax(axis=1) + 1
df = pd.DataFrame(np.array(ans))
df.to_csv('ans.csv', index=False)
