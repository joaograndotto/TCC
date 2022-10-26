from flask import Flask, request, render_template
import os 
import pandas as pd
import pickle


app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

#carregar modelo
def classify_utterance(utt):
    # load the model
    loaded_model = pickle.load(open('models\ifgIASigmond.model', 'rb'))
    # make a prediction
    resultado = loaded_model.predict(utt)
    return resultado

UPLOAD_FOLDER = 'static/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#request pelo arquivo csv 
@app.route("/request_csv", methods=['POST'])
def uploadFiles():
    #upload do arquivo
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        #caminho para o upload do arquivo
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        #salvando arquivo
        uploaded_file.save(file_path)

    #lendo csv não classificado
    dataset = pd.read_csv(file_path, sep=',', encoding='utf-8')
    predict = classify_utterance(dataset.squeeze())
    
    dataset["afiliacao_classificada"] = predict
    dataset['afiliacao_classificada'] = dataset['afiliacao_classificada'].apply(lambda x: "Instituto Federal de Goiás" if x == 1 else 'Não é Instituto Federal de Goiás')
    #Salvando o arquivo classificado localmente
    dataset.to_csv("static/arquivo_classificado.csv", index=False)
    return ''' <h3><a href="static/arquivo_classificado.csv" download>Clique Aqui para fazer o download</a></h3>''' #return para o download do arquivo classificado


#request pelo html
@app.route('/request_input', methods=['GET', 'POST'])
def request_input():
    affiliation = request.form.get('affiliation')
    affiliation = [affiliation]
    result = classify_utterance(affiliation)
    r = ['Não é Instituto Federal de Goiás','É Instituto Federal de Goiás']
    return '''
                  <h1>A afiliação é: {}</h1>'''.format(r[result[0]])
if __name__ == '__main__':
    app.run(debug=True)