from flask import Flask, request
import os 
import pandas as pd
import pickle



app = Flask(__name__)

#carregar modelo
def classify_utterance(utt):
    # load the model
    loaded_model = pickle.load(open('models/fiocruz.novaia.model', 'rb'))
    # make a prediction
    resultado = loaded_model.predict(utt)
    return resultado


UPLOAD_FOLDER = 'static/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#request pelo arquivo csv 
@app.route("/uploadcsv", methods=['POST'])
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
      predict = classify_utterance(dataset['Affiliations'])
      # Transformando a lista(numpy) em uma dataframe
      dataset_classificado = pd.DataFrame (predict, columns = ['Affiliations'])
      #Salvando o arquivo classificado localmente
      dataset_classificado.to_csv("static/Classificado.csv", index=False)
      return dataset_classificado.to_json(orient="split", index=False, force_ascii=False) #return em json
      
if __name__ == '__main__':
    app.run(debug=True)