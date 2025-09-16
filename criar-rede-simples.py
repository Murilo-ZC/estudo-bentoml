import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# Garante que os dados aleatórios são gerados com o mesmo seed
np.random.seed(42)

# Dados carregados da etapa anterior
df = pd.read_csv("dados.csv", skiprows=[1,2], header=0)

# Fazendo a separação das partes que serão utilizadas do df
dados_df = df[['Ticker','BTC-USD','DOGE-USD']]

# Ajustando o nome das colunas
dados_df.columns = ['Date', 'BTC', 'DOGE']

# Ajustando o tipo dos dados de data para Datetime
dados_df["Date"] = pd.to_datetime(dados_df["Date"], format="%Y-%m-%d", errors="coerce")

# Função para, dada uma data, criar features que possam ser elementos de entrada
# A tipagem indica que vamos receber apenas uma coluna de dados e retornar um dataframe
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

# Cria a representação dos dados entre entrada (X) e saída (y)
X = make_date_features(dados_df['Date'])
y_btc = dados_df['BTC']
y_doge = dados_df['DOGE']

print(dados_df.head(1))

# Faz a separação dos dados entre treino e validação. Por ser dados temporais, não fazemos o embaralhamento
X_train,X_test,y_train,y_test = train_test_split(
    X,y_btc, test_size=0.2, shuffle=False
)

# Cria o primeiro Pipeline para BTC
pipe_btc = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(128,8),
        activation='relu',
        solver='adam',
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=10000,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42
    ))
])

# Treina o modelo para BTC
pipe_btc.fit(X_train,y_train)


# Testa as métricas do modelo treinado
pred_btc = pipe_btc.predict(X_test)
r2_total = r2_score(y_test, pred_btc)
mae_y_btc = mean_absolute_error(y_test, pred_btc)

print("===============BTC=======================")
print(f"R² médio (y BTC): {r2_total:.3f}")
print(f"MAE y BTC: {mae_y_btc:.3f}")
data = "2025-09-20"
X_new = make_date_features([data])
y_pred_new = pipe_btc.predict(X_new)
print(f"Previsão para {data}: y={y_pred_new[0]:.3f}")

# -------------------------------------------------------
# Faz a separação dos dados entre treino e validação. Por ser dados temporais, não fazemos o embaralhamento
X_train,X_test,y_train,y_test = train_test_split(
    X,y_doge, test_size=0.2, shuffle=False
)
# Cria o segundo Pipeline para DOGE
pipe_doge = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(64,128),
        activation='relu',
        solver='adam',
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=10000,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42
    ))
])

# Treina o modelo para DOGE
pipe_doge.fit(X_train,y_train)


# Testa as métricas do modelo treinado
pred_doge = pipe_doge.predict(X_test)
r2_total = r2_score(y_test, pred_doge)
mae_y_doge = mean_absolute_error(y_test, pred_doge)

print("===============DOGE=======================")
print(f"R² médio (y DOGE): {r2_total:.3f}")
print(f"MAE y DOGE: {mae_y_doge:.3f}")
data = "2025-09-20"
X_new = make_date_features([data])
y_pred_new = pipe_doge.predict(X_new)
print(f"Previsão para {data}: y={y_pred_new[0]:.3f}")

# Salvando os modelos treinados
joblib.dump(pipe_btc, "modelo_btc.joblib")
joblib.dump(pipe_doge, "modelo_doge.joblib")