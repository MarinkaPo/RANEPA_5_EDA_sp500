import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yfinance as yf

st.title('S&P 500 App')

st.markdown("""
Приложение выводит топ-500 компаний индекса **S&P 500** (из Википедии) и соответствующие им **цены закрытия** (по годам)!
* **Библиотеки Python:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Источник:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
""")

st.sidebar.header('Выберите характеристики')

# Смотрим данные:
#
@st.cache
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header = 0)
    df = html[0]
    return df

df = load_data()
sector = df.groupby('GICS Sector')

# Боковая панель - Выбор профиля компании
sorted_sector_unique = sorted( df['GICS Sector'].unique() )
selected_sector = st.sidebar.multiselect('Выбор профиля компании:', sorted_sector_unique, sorted_sector_unique)

# Фильтрация
df_selected_sector = df[ (df['GICS Sector'].isin(selected_sector)) ]
df_selected_sector = df_selected_sector.astype(str)

st.header('Выбранные компании:')
st.write('Размер таблицы: ' + str(df_selected_sector.shape[0]) + ' строк и ' + str(df_selected_sector.shape[1]) + ' колонок.')
st.dataframe(df_selected_sector)

# Загружаем данные
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Загрузить CSV-файл</a>'
    return href

st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)

# https://pypi.org/project/yfinance/

data = yf.download(
        tickers = list(df_selected_sector[:10].Symbol),
        period = "ytd",
        interval = "1d",
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
        threads = True,
        proxy = None
    )

# Plot Closing Price of Query Symbol
def price_plot(symbol):
  df = pd.DataFrame(data[symbol].Close)
  df['Date'] = df.index
  f, ax = plt.subplots(figsize=(10, 7))
  plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
  ax = plt.plot(df.Date, df.Close, color='black', alpha=0.8)
  plt.xticks(rotation=90)
  plt.title(symbol, fontweight='bold')
  plt.xlabel('Дата', fontweight='bold')
  plt.ylabel('Цена закрытия', fontweight='bold')
  return st.pyplot(f)

num_company = st.sidebar.slider('Кол-во компаний', 1, 5)

if st.button('Смотреть графики'):
    st.header('Цена закрытия')
    for i in list(df_selected_sector.Symbol)[:num_company]:
        price_plot(i)
