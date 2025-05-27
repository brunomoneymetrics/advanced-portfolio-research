import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from cycler import cycler
import plotnine as p9
import seaborn as sns
from scipy.stats import norm


#---------------------

plt.style.use('fivethirtyeight')


# Sua paleta: verde-escuro e dourado queimado
custom_colors = ['#264653', '#E9C46A']

# Aplicar como padrão global no notebook
plt.rcParams['axes.prop_cycle'] = cycler(color=custom_colors)

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = '#CCCCCC'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12



# ----- função scatter plot 



def plot_scatter_with_reference_lines(
    data,
    x_col,
    y_col,
    stat='mean',  # ou 'median'
    figsize=(10, 10),
    x_label=None,
    y_label=None,
    title=None,
    subtitle=None,
    caption=None,
    line_color='red',
    line_style='--',
    line_width=2
):
    # Verificação de colunas
    if x_col not in data.columns or y_col not in data.columns:
        raise ValueError("As colunas especificadas não estão no DataFrame.")
    
    # Cálculo das estatísticas
    if stat == 'mean':
        x_stat = data[x_col].mean()
        y_stat = data[y_col].mean()
    elif stat == 'median':
        x_stat = data[x_col].median()
        y_stat = data[y_col].median()
    else:
        raise ValueError("O parâmetro 'stat' deve ser 'mean' ou 'median'.")

    # Criação da figura
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(data[x_col], data[y_col])

    # Linhas de referência
    ax.axvline(x_stat, color=line_color, linestyle=line_style, linewidth=line_width)
    ax.axhline(y_stat, color=line_color, linestyle=line_style, linewidth=line_width)

    # Rótulos e título
    ax.set_xlabel(x_label if x_label else x_col)
    ax.set_ylabel(y_label if y_label else y_col)
    
    if subtitle:
        ax.set_title(subtitle, fontsize=12)

    if title:
        plt.suptitle(title, fontsize=16, y=1.02)

    if caption:
        fig.text(0.9, 0.01, caption, ha='right', fontsize=9, style='italic')

    plt.grid(True)
    plt.tight_layout()
    plt.show()


# volatilidade ativo individual ---- 
def calcular_volatilidades(NMV, Beta, Std_market, Std_resid):
    """
    Calcula:
    - Volatilidade do termo de mercado
    - Volatilidade idiossincrática
    - Volatilidade total dos retornos do portfólio

    Parâmetros:
    - NMV: Net Market Value (float)
    - Beta: sensibilidade da ação ao mercado (float)
    - Std_market: volatilidade do mercado (float)
    - Std_resid: volatilidade idiossincrática (resíduo) da ação (float)

    Retorna:
    - vol_mercado: volatilidade do termo de mercado (float)
    - vol_idios: volatilidade idiossincrática (float)
    - vol_total: volatilidade total (float)
    """

    vol_mercado = NMV * Beta * Std_market
    vol_idios = NMV * Std_resid

    # Variância total é a soma das variâncias (independência dos componentes)
    var_total = vol_mercado**2 + vol_idios**2
    vol_total = np.sqrt(var_total)

    return vol_mercado, vol_idios, vol_total


# -- volatilidade da carteira --- 

def calcular_volatilidade_carteira(ativos, market_vol):
    """
    Calcula a volatilidade total de uma carteira, decompondo entre
    contribuição de mercado e contribuição idiossincrática.

    Parâmetros:
    - ativos: dicionário com cada ativo contendo:
        - 'beta': beta do ativo
        - 'idio_vol': volatilidade idiossincrática diária (em %)
        - 'market_value': valor de mercado (em milhões de dólares)
    - market_vol: volatilidade diária do mercado (em %)

    Retorna:
    - dicionário com beta em dólares, volatilidade de mercado, idiossincrática e total (em milhares de dólares)
    """

    # 1 e 2: calcular beta em dólares da carteira
    dollar_betas = [v['beta'] * v['market_value'] for v in ativos.values()]
    beta_dolar_carteira = sum(dollar_betas)

    # 3: contribuição da volatilidade de mercado
    volatilidade_mercado = beta_dolar_carteira * (market_vol / 100)

    # 4: contribuição da volatilidade idiossincrática
    soma_quadrados_idio = sum((v['market_value'] * v['idio_vol'])**2 for v in ativos.values())
    volatilidade_idio = np.sqrt(soma_quadrados_idio)

    # 5: volatilidade total da carteira
    volatilidade_total = np.sqrt(volatilidade_mercado**2 + volatilidade_idio**2)

    return {
        'beta_dolar': beta_dolar_carteira,
        'vol_mercado': volatilidade_mercado  ,      # em mil dólares
        'vol_idiossincratica': volatilidade_idio ,   # em mil dólares
        'vol_total': volatilidade_total              # em mil dólares
    }



# --- obtenção de metricas 

def obter_metricas_ativos(dados, tickers_ativos, ticker_mercado, market_caps=None):
    """
    Calcula beta, desvio padrão idiossincrático e valor de mercado (se disponível) para múltiplos ativos.

    Parâmetros:
    - dados: DataFrame com colunas de retornos (ex: 'AAPL_returns', 'SP500_returns')
    - tickers_ativos: lista de colunas dos ativos (ex: ['AAPL_returns', 'MSFT_returns'])
    - ticker_mercado: nome da coluna do mercado (ex: 'SP500_returns')
    - market_caps: dicionário {ticker: valor_em_milhoes}, opcional. Se não for fornecido, assume 100 milhões.

    Retorna:
    - dicionário {ticker: {'beta': ..., 'idio_vol': ..., 'market_value': ...} }
    """
    resultados = {}

    for ticker in tickers_ativos:
        # Regressão linear simples: ativo ~ mercado
        formula = f"{ticker} ~ {ticker_mercado}"
        modelo = smf.ols(formula, data=dados).fit()

        beta = modelo.params[ticker_mercado]
        idio_vol = np.std(modelo.resid)

        # Define valor de mercado (em milhões)
        if market_caps and ticker in market_caps:
            market_value = market_caps[ticker]
        else:
            market_value = 100000  # valor default: 100 milhões

        resultados[ticker] = {
            'beta': beta,
            'idio_vol': idio_vol,
            'market_value': market_value
        }

    return resultados




#------ retornos -----------
def create_cumulative_returns(tickers, start, end):
    # Baixa os dados
    precos_ativos = yf.download(tickers, start=start, end=end, ignore_tz=True)
    precos_ativos = precos_ativos.loc[:, ('Close', slice(None))]
    precos_ativos.columns = sorted(tickers)

    # Calculando os retornos diários
    daily_returns = precos_ativos[tickers].pct_change().dropna()

    # Calculando o retorno acumulado
    cumulative_returns = (1 + daily_returns).cumprod()
    cumulative_returns = cumulative_returns.iloc[-1] - 1

    return cumulative_returns



# --------- grafico barras 




def plot_retornos_acumulados(
    df,
    x_col='Ações',
    y_col='Retornos',
    titulo='Retornos Acumulados de Ações',
    subtitulo='Ranking das 5 Maiores Altas e Quedas de Ações – 2024 a 2025 (até 25/05/2025)',
    legenda=False,
    cores=['#264653', '#E9C46A'],
    autor='Bruno Araújo',
    fonte='Yahoo Finance',
    figura_tamanho=(10, 6),
    tamanho_label=10
):
    """
    Cria um gráfico de barras horizontais com os retornos acumulados usando plotnine.

    Parâmetros:
        df (DataFrame): Dados contendo as colunas de ações e retornos.
        x_col (str): Nome da coluna com os nomes das ações.
        y_col (str): Nome da coluna com os retornos.
        titulo (str): Título do gráfico.
        subtitulo (str): Subtítulo do gráfico.
        legenda (bool): Exibir legenda.
        cores (list): Lista com duas cores [negativo, positivo].
        autor (str): Nome do autor para o caption.
        fonte (str): Fonte dos dados para o caption.
        figura_tamanho (tuple): Tamanho da figura (largura, altura).
        tamanho_label (int): Tamanho do texto das labels.
    """

    df = df.copy()
    df['positivo'] = df[y_col] > 0
    df['label'] = (df[y_col] * 100).apply(lambda x: f"{x:.2f}%")

    plot = (
        p9.ggplot(df, p9.aes(x=f'reorder({x_col}, {y_col})', y=y_col, fill='positivo'))
        + p9.geom_col(show_legend=legenda)
        + p9.geom_label(p9.aes(label='label'), size=tamanho_label, fill='white')
        + p9.scale_fill_manual(values=cores)
        + p9.scale_y_continuous(labels=lambda l: ["{:,.0f}%".format(v * 100) for v in l])
        + p9.labs(
            title=titulo,
            subtitle=subtitulo,
            x="",
            y="%",
            caption=f"Elaborado por {autor} com dados do {fonte}"
        )
        + p9.theme_minimal()
        + p9.theme(figure_size=figura_tamanho)
        + p9.coord_flip()
    )

    return plot


# Gráfico da Normal 



def plot_risco_normal(vol_mercado, vol_idios, vol_total):
    """
    Plota a distribuição normal das variações monetárias com base na volatilidade total,
    e destaca as perdas esperadas em R$.

    Parâmetros:
    - vol_mercado: volatilidade sistemática em R$
    - vol_idios: volatilidade idiossincrática em R$
    - vol_total: volatilidade total em R$ (sqrt(vol_mercado² + vol_idios²))
    """

    # Dados da curva normal (em R$)
    x = np.linspace(-4 * vol_total, 4 * vol_total, 1000)
    y = norm.pdf(x, 0, vol_total)

    # Plotagem
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Distribuição Normal dos Retornos em R$', color='navy')

    # Faixas de 1σ e 2σ
    for i, alpha in zip([1, 2], [0.2, 0.1]):
        plt.fill_between(
            x,
            0,
            y,
            where=(x >= -i * vol_total) & (x <= i * vol_total),
            color='red',
            alpha=alpha,
            label=f'{i}σ ≈ {norm.cdf(i)-norm.cdf(-i):.0%} de confiança'
        )

    # Linha de perda esperada (1σ)
    perda_1dp = -vol_total
    plt.axvline(perda_1dp, color='black', linestyle='--',
                label=f'Perda esperada (1σ): R$ {abs(perda_1dp):,.2f}')

    # Título e eixos
    plt.title('Distribuição Normal dos Retornos (Variação em R$)', fontsize=14)
    plt.xlabel('Variação em R$')
    plt.ylabel('Densidade de Probabilidade')
    plt.annotate(
    'Elaborado por Bruno Araújo',
    xy=(0.95, 0.02),  
    xycoords='figure fraction',
    ha='right',
    fontsize=9,
    style='italic')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
   
    plt.show()




