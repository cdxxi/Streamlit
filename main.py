import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

plt.style.use('seaborn-v0_8')


st.title('Прогнозирование временных рядов')

#Загрузка файла
uploaded_file = st.file_uploader("Загрузите ваш CSV или Excel файл с временным рядом", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Определяем, является ли файл CSV или Excel
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        data = pd.read_excel(uploaded_file)
    else:
        st.error("Неверный формат файла. Пожалуйста, загрузите CSV или Excel файл.")
        st.stop() 
    st.write('Ваш файл', data)

    date_column = st.sidebar.selectbox('Выберите колонку с датой', data.columns)
    target_column = st.sidebar.selectbox('Выберите колонку с целевой переменной', data.columns)

    if date_column == target_column:
        st.error("Колонка с датой и целевой переменной должны быть разными.")
        st.stop()

    #Модернизация датасета в нужный
    data[date_column] = pd.to_datetime(data[date_column])
    data.set_index(date_column, inplace=True)
    df = pd.DataFrame(data[target_column])


    dfw = df[target_column].resample('W').mean()
    dfws = df[target_column].resample('W').sum()
    dfms = df[target_column].resample('M').sum()

    df = pd.DataFrame(dfws, columns=['Sales'])

    #Графики
    st.subheader('Графики средних/суммы продаж понедельно')
    fig, axes = plt.subplots(2,1, figsize=(30,30))
    ax1 = sns.lineplot(x=dfw.index, y=dfw.values, ax=axes[0])
    ax1.set_title('Неделя/среднее')
    ax1.set_ylabel('Sales')
    ax2 = sns.lineplot(x=dfws.index, y=dfws.values, ax=axes[1])
    ax2.set_title('Неделя/сумма')
    ax2.set_ylabel('Sales')
    st.pyplot(plt)

    n_window = 30

    moving_average_pred = df[target_column].rolling(window=n_window, closed='left', min_periods=1).mean()

    st.subheader('График скользящего среднего')
    plt.figure(figsize=(20,10))
    plt.plot(df[target_column], label='real')
    plt.plot(moving_average_pred, label='pred')
    plt.xticks(dfms.index, rotation=45)
    plt.legend()
    st.pyplot(plt)

    st.subheader('Декомпозиция')
    st.write(':blue[_Тренд_] отражает долгосрочное направление изменения данных. Он показывает, увеличиваются или уменьшаются продажи в среднем за рассматриваемый период.')
    st.write(':green[_Сезонность_] отражает периодические колебания в данных, которые повторяются через определенные промежутки времени. Это может быть связано с сезонными изменениями в спросе, например, повышенные продажи в праздничные периоды.')
    st.write(':red[_Остатки_] — это та часть данных, которая не объясняется трендом и сезонностью. Они представляют собой случайные колебания или шум в данных.')
    fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1, figsize=(15,8))
    res = seasonal_decompose(df[target_column])
    res.observed.plot(ax=ax1, c='black')
    ax1.set_title('Наблюдаемые данные', loc='right')
    res.trend.plot(ax=ax2, c='blue')
    ax2.set_title('Тренд', loc='right')
    res.seasonal.plot(ax=ax3, c='green')
    ax3.set_title('Сезонность', loc='right')
    res.resid.plot(ax=ax4, c='red')
    ax4.set_title('Остатки', loc='right')
    st.pyplot(plt)
    
    st.subheader('График прогноза модели Prophet')
    data_prophet = df[target_column].reset_index().rename(columns={date_column: 'ds', target_column: 'y'}) # Обязательное имена колнок с датой и таргетотм в prophet
    data_train = data_prophet[data_prophet['ds'].dt.year < 2017]
    data_test = data_prophet[data_prophet['ds'].dt.year >= 2017]
    model = Prophet()
    model.fit(data_train)

    seasonality_period = 12 
    number_of_future_predicted_points = 5 * seasonality_period

    future = model.make_future_dataframe(periods=number_of_future_predicted_points, freq='M')
    forecast = model.predict(future)
    forecast_train = forecast[:-number_of_future_predicted_points]
    forecast_test = forecast[-number_of_future_predicted_points: -number_of_future_predicted_points + len(data_test)]
    forecast_future = forecast[-number_of_future_predicted_points + len(data_test):]


    prophet_mae_train = np.round(mean_absolute_error(data_train['y'], forecast_train['yhat']), 1)
    prophet_mae_test = np.round(mean_absolute_error(data_test['y'], forecast_test['yhat']), 1)

    plt.figure(figsize=(20, 10))
    plt.plot(df[target_column], label='true_data', marker='o')

    plt.plot(forecast_train['ds'], forecast_train['yhat'], marker='v', linestyle=':', label=f'forecast_train, mae={prophet_mae_train}')
    plt.plot(forecast_test['ds'], forecast_test['yhat'], marker='v', linestyle=':', label=f'forecast_test = mae={prophet_mae_test}')
    plt.plot(forecast_future['ds'], forecast_future['yhat'], marker='v', linestyle=':', label='forecast_future', color='b')

    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.15)
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)

