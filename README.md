# Utilizando o BentoML

Primeiro, algumas coisas que precisamos fazer:
1. Criar um `venv` para colocar as dependencias
2. Instalar as dependencias
3. Criar e treinar um modelo
4. Encapsular o modelo com o BentoML para servir ele
5. Realizar o deploy do modelo com o BentoML

Para consultar a documentação:
- [Documentação](https://github.com/bentoml/BentoML)

## 1. Criar um `venv` para colocar as dependencias

Primeiro vamos criar nosso ambiente virtual de Python. Esse passo é interessante para realizar uma separação da nossa instalação de Python dos projetos desenvolvidos. Uma das vantagens que conseguimos com esse procedimento, é isolar as dependencias do nosso projeto com as demais no sistema. Existem diversas ferramentas que podem fazer isso, minha recomendação é iniciar esse processo com o pacote `venv` do Python.

Para criar o `venv`, cada sistema operacional tem seu respectivo formato, recomendo ver a documentação. Aqui vai um exemplo para os alguns OS:

```sh
# Windows - Vai criar um diretório chamado venv na pasta atual
python -m venv venv
# Ativar o ambiente virtual de Python
.\venv\Scripts\activate
```

```sh
# Linux e MacOS
python3 -m venv venv
./venv/bin/activate
```

## 2. Instalar as dependencias

Agora das dependencias, vamos instalar algumas para que o BentoML possa funcionar e depois vamos fazer as do projeto.

```sh
# Linux Mac
python3 -m pip install bentoml
# Windows
# python -m pip install bentoml
```

Boa agora vamos para nossa aplicação!

## 3. Criar e treinar um modelo

Vamos criar nosso modelo para prever o valor de um criptoativo, utilizando uma rede neural de várias camadas. Primeiro vamos pegar os dados de um ativo utilizando o Yahoo Finance ([doc](https://pypi.org/project/yfinance/)).

Primeiro vamos instalar essa dependencia:

```sh
# Linux Mac
python3 -m pip install yfinance
# Windows
# python -m pip install yfinance
```

Além das dependencias, vamos instalar algumas ferramentas auxiliares:

```sh
# Linux Mac
python3 -m pip install pandas scikit-learn
# Windows
# python -m pip install pandas scikit-learn
```

Vamos testar nosso sistema vendo se conseguimos pegar algumas informações sobre o valor de um criptoativo. 

```python
import yfinance as yf
import pandas as pd

# Coletando dado do BTC-USD
dados = yf.download(
    tickers=["DOGE-USD","BTC-USD"],
    period="2y",
    interval="1d"
)

# Salvando os dados em um CSV
dados.to_csv("dados.csv")
```

Os dados foram baixados de `DOGE` e `BTC`, vamos utilizar apenas o valor da data e do `BTC`, esse exemplo foi só para ver como baixar mais dados de mais elementos. Vamos agora criar um modelo de RNN com o SKLearn. A documentação para realizar regressões pode ser vista [aqui](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html).

Vamos agora trabalhar na criação dos modelos. Dentre as manipulações que vamos fazer uma delas é a representação cíclica para as datas. Essa representação traz uma vantagem em relação a abordagem linear, pois, meses como Dezembro e Janeiro estão próximos, mas como um é o mês 12 e o outro é o mês 1, parece que estão bastante distantes. Uma solução para isso é representar eles utilizando sua posição em um plano cartesiano com as funções trigonométricas seno e cosseno. Um pouco mais de detalhe aqui:

### O Problema da Representação Linear

Quando você tem um recurso cíclico, como o mês do ano, e o representa com números simples (1 a 12), o modelo de machine learning pode interpretar mal a relação entre os dados.
- O modelo vê a distância entre mês 1 (janeiro) e mês 2 (fevereiro) como 1.
- Ele vê a distância entre mês 12 (dezembro) e mês 1 (janeiro) como 11.
Isso é um problema, pois na realidade, o mês de dezembro está "ao lado" do mês de janeiro, e essa representação linear não captura essa proximidade, criando uma descontinuidade artificial.

### A Solução: Usando Seno e Cosseno

A codificação cíclica resolve isso transformando a variável temporal em duas novas variáveis usando as funções trigonométricas seno e cosseno. A ideia é mapear o tempo em um círculo. Imagine que o mês de janeiro está na posição de 30° em um círculo, fevereiro em 60°, e assim por diante. Cada ponto no círculo é representado por duas coordenadas (x, y), que são o cosseno e o seno do ângulo, respectivamente.

As fórmulas são:

$$
\text{seno} = \sin\left(\frac{2 \pi \cdot \text{valor\_original}}{\text{valor\_maximo}}\right)
$$
$$
\text{cosseno} = \cos\left(\frac{2 \pi \cdot \text{valor\_original}}{\text{valor\_maximo}}\right)
$$

Com essa transformação:

- O mês 12 (dezembro) e o mês 1 (janeiro) estarão próximos um do outro no plano cartesiano.
- O mês 6 (junho) estará no lado oposto do círculo, refletindo sua distância real de janeiro e dezembro.

Isso permite que o modelo de machine learning entenda que as datas no início e no fim do ciclo estão, na verdade, muito próximas.

Para criar nossos modelos de rede:

```python
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
```

Boa, agora nós temos um modelo e conseguimos utilizar ele! Vamos agora carregar ele para adicionar o BentoML.

## 4. Encapsular o modelo com o BentoML para servir ele

Primeiro vamos carregar nosso modelo e ver se ele continua funcionando. Vamos fazer isso com o arquivo `roda-rede.py`.

```python
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
```

Beleza, nosso sistema está funcionando carregando os modelos, mas vamos ligá-lo ao BentoML. O fluxo do BentoML é:
- Carregamos nosso modelo para o BentoML. Ele gerencia os modelos localmente para poder servir diferentes verções;
- O modelo é carregado com o BentoML. O modelo é carregado dos arquivos salvos por ele;
- O modelo é servido como aplicação. O modelo fica disponível para ser acessado como um serviço servido via HTTP.

Mais detalhes podem ser vistos [aqui](https://docs.bentoml.com/en/latest/build-with-bentoml/model-loading-and-management.html). As ferramentas que podem ser utilizadas para salvar os modelos dentro do BentoML podem ser vistas [aqui](https://docs.bentoml.com/en/latest/reference/bentoml/frameworks/index.html). Como no nosso caso, vamos utilizar um modelo que foi treinado com o SKLearn, vamos utilizar [essa](https://docs.bentoml.com/en/latest/reference/bentoml/frameworks/sklearn.html) referencia da documentação.

Vamos salvar o modelo no arquivo `salva-bentoml.py`:

```python
import joblib
import bentoml

# Carregando o modelo
pipe_btc = joblib.load("modelo_btc.joblib")
pipe_doge = joblib.load("modelo_doge.joblib")

bento_model_btc = bentoml.sklearn.save_model('btc', pipe_btc)
print(bento_model_btc)
bento_model_doge = bentoml.sklearn.save_model('doge', pipe_doge)
print(bento_model_doge)
```

Ao executar esse script, vamos ver ambos os modelos salvos e uma tag associada a eles será apresentada ao usuário. Podemos ver todos os modelos que estão salvos utilizando o comando: `python3 -m bentoml models list`.

Agora vamos carregar e utilizar nossos modelos. Para isso vamos utilizar o `service.py`. Esse é o arquivo default que o BentoML busca quando utilizamos o comando `python3 -m bentoml serve .`. Importante, antes precisamos de um arquivo com as bibliotecas que vamos utilizar:

```bash
python3 -m pip freeze > requirements.txt
```

```python
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
```

Agora podemos testar nossos modelos em `http://localhost:3000`. Beleza! Temos nosso sistema rodando! Vamos agora criar um container que possa rodar nossa aplicação.

## 5. Realizar o deploy do modelo com o BentoML

Vamos agora criar um container para rodar nossa aplicação. Precisamos primeiro construir a imagem com:

```bash
python3 -m bentoml build
```

Agora vamos criar nossa imagem:

```bash
python3 -m bentoml containerize ModelosMLPCripto:latest -t mlp-cripto:1.0
```

E para rodar nossa imagem:

```bash
docker run --rm -p 3000:3000 mlp-cripto:1.0
```

Pronto! Nosso modelo está sendo executado agora!