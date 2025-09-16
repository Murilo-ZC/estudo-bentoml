import joblib
import bentoml

# Carregando o modelo
pipe_btc = joblib.load("modelo_btc.joblib")
pipe_doge = joblib.load("modelo_doge.joblib")

bento_model_btc = bentoml.sklearn.save_model(
    'btc', pipe_btc
    )
print(bento_model_btc)
bento_model_doge = bentoml.sklearn.save_model(
    'doge', pipe_doge
    )
print(bento_model_doge)