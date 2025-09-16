import bentoml
from bentoml import api, service
from bentoml.models import BentoModel
import pandas as pd
import numpy as np
from bentoml.images import Image

# Função para gerar features da data
def make_date_features(s: any) -> pd.DataFrame:
    s = pd.Series(s)
    # Caso a série não seja do tipo Datetime
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s)
    # Cria um dataframe para guardar as features criadas. Utilizar os mesmos indices que o df enviado
    df_feat = pd.DataFrame(index=s.index)
    # Gerando os componentes básicos
    df_feat['year'] = s.dt.year
    df_feat['month'] = s.dt.month
    df_feat['day'] = s.dt.day
    df_feat['dayofweek'] = s.dt.day_of_week
    df_feat['dayofyear'] = s.dt.day_of_year

    # Semana ISO - semana do ano referente a norma ISO 8601. Cada semana do ano tem um número pela norma
    df_feat['week'] = (s.dt.isocalendar()).week.astype(int)

    # Representação cíclica
    df_feat['month_sin'] = np.sin(2*np.pi * df_feat['month']/12)
    df_feat['month_cos'] = np.cos(2*np.pi * df_feat['month']/12)
    df_feat['dow_sin'] = np.sin(2*np.pi * df_feat['dayofweek']/7)
    df_feat['dow_cos'] = np.cos(2*np.pi * df_feat['dayofweek']/7)
    df_feat['doy_sin'] = np.sin(2*np.pi * df_feat['dayofyear']/365.25)
    df_feat['doy_cos'] = np.cos(2*np.pi * df_feat['dayofyear']/365.25)

    return df_feat



@service(
    image=Image(python_version="3.11").requirements_file("./requirements.txt")
)
class ModelosMLPCripto:
    # Pega os modelos do store
    btc_ref = BentoModel("btc:latest")
    doge_ref = BentoModel("doge:latest")
    # Carregando os modelos
    def __init__(self) -> None:
        self.model_btc = bentoml.sklearn.load_model(self.btc_ref)
        self.model_doge = bentoml.sklearn.load_model(self.doge_ref)

    @api(batchable=False)
    def predict_btc(self, date : str):
        ndate = make_date_features(date)
        return {"prediction":self.model_btc.predict(ndate)[0]}
    
    @api(batchable=False)
    def predict_doge(self, date : str):
        ndate = make_date_features(date)
        return {"prediction":self.model_doge.predict(ndate)[0]}