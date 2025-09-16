import joblib
import pandas as pd
import numpy as np
import joblib

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

# Carregando o modelo
pipe_btc = joblib.load("modelo_btc.joblib")
pipe_doge = joblib.load("modelo_doge.joblib")

# Utilizando os modelos
data = "2025-09-30"
X_new = make_date_features([data])
y_pred_new = pipe_btc.predict(X_new)
print(f"Previsão para {data} - BTC: y={y_pred_new[0]:.3f}")
y_pred_new = pipe_doge.predict(X_new)
print(f"Previsão para {data} - DOGE: y={y_pred_new[0]:.3f}")