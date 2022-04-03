from flask import Flask, request, jsonify, render_template
from dummy import output
import pandas as pd
app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/resources',methods=['POST'])
def resources():
     if request.method == 'POST':
        age = float(request.form.get("age"))
        gap = float(request.form.get("gap"))
        exp = float(request.form.get("exp"))
        domain = request.form.get("domain")
        state = request.form.get("state")
        salary = float(request.form.get("salary"))
        #result = "abc"
        l = [0,0,0,0,0,0,0,0]
        l.insert(0,age)
        l.insert(1,gap)
        l.insert(2,exp)
        l.insert(3,salary)
        if(domain == "hr"):
            l[4] = 1
        if(domain == "it"):
            l[5] = 1
        if(domain == "id"):
            l[6] = 1
        if(domain == "leg"):
            l[7] = 1
        if(domain == "medc"):
            l[8] = 1
        if(domain == "med"):
            l[9] = 1
        if(domain == "pm"):
            l[10] = 1
        if(domain == "ps"):
            l[11] = 1



        result = output(l)
        #result = output(l[0],l[1],l[2],l[3],l[4],l[5],l[6],l[7],l[8],l[9],l[10],l[11])
        print(result[0])
        
        lis = links(result[0])  


        return render_template('resources.html', lis=lis)

def links(role):
    df = pd.read_csv('resourcessheet.csv')
    recommendations = set()
    for r in df["JOB DESCRIPTION"]:
        s = r.replace(" ", "").lower()
        if role.lower() in s:
            x = df.index[df['JOB DESCRIPTION']==r].tolist()
            #print(df.index[df['JOB DESCRIPTION']==r].tolist())
            for i in x:
                y = df.iloc[i]["LINK"]
                recommendations.add(y)
    return recommendations

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)