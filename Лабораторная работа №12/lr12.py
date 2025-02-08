import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['start_time'])
    df['usage_seconds'] = df['usage_time'].apply(lambda x: sum(int(t) * 60**i for i, t in enumerate(reversed(str(x).split(':')))))
    df['hour'] = df['start_time'].dt.hour
    df['date'] = df['start_time'].dt.date
    df['month'] = df['start_time'].dt.to_period('M')
    return df

def most_frequent_time(df):
    return df['hour'].value_counts().idxmax()

def ip_change_frequency(df):
    return df.groupby('name')['IP'].nunique()

def device_change_frequency(df):
    return df.groupby('name')['MAC'].nunique()

def average_usage(df):
    hourly_avg = df.groupby('hour')['usage_seconds'].mean()
    daily_avg = df.groupby('date')['usage_seconds'].sum().mean()
    monthly_avg = df.groupby('month')['usage_seconds'].sum().mean()
    return hourly_avg, daily_avg, monthly_avg

def shared_devices(df):
    return df.groupby('MAC')['name'].nunique().gt(1).sum()

filepath = 'internet_session.csv'
df = load_data(filepath)
    
print("Наиболее часто используемое время суток:", most_frequent_time(df))
print("Частота смены IP:")
print(ip_change_frequency(df))
print("Частота смены устройства:")
print(device_change_frequency(df))
hourly_avg, daily_avg, monthly_avg = average_usage(df)
print(f"Среднее использование: {hourly_avg.mean():.2f} сек/час, {daily_avg:.2f} сек/день, {monthly_avg:.2f} сек/месяц")
print("Количество устройств, использованных разными пользователями:", shared_devices(df))
    
plt.figure(figsize=(10, 5))
hourly_avg.plot(kind='bar', title='Среднее использование Интернета по часам')
plt.xlabel('Час')
plt.ylabel('Среднее время использования (сек)')
plt.show()
