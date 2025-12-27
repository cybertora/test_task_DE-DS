import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

summary = pd.read_csv('csvs/Measurement_summary.csv')
stations = pd.read_csv('csvs/Measurement_station_info.csv')
items = pd.read_csv('csvs/Measurement_item_info.csv')

summary['Measurement date'] = pd.to_datetime(summary['Measurement date'])
summary['date'] = summary['Measurement date'].dt.date
summary['hour'] = summary['Measurement date'].dt.hour
summary['weekday'] = summary['Measurement date'].dt.weekday
summary['is_weekend'] = summary['weekday'].isin([5, 6]).astype(int)
summary['month'] = summary['Measurement date'].dt.month
summary['season'] = ((summary['month'] % 12 + 3) // 3)  # 1=Зима, 2=Весна, 3=Лето, 4=Осень

print("Данные загружены успешно!")
print(f"Размер summary: {summary.shape}")
print("Колонки:", summary.columns.tolist())

# 1. Будни vs Выходные (берем PM2.5 как основной показатель)
print("\n1. Загрязненность в будни и выходные (по PM2.5)")

pm25_data = summary[['is_weekend', 'PM2.5']].dropna()

weekday_pm25 = pm25_data[pm25_data['is_weekend'] == 0]['PM2.5']
weekend_pm25 = pm25_data[pm25_data['is_weekend'] == 1]['PM2.5']

print(f"Среднее в будни: {weekday_pm25.mean():.2f}")
print(f"Среднее в выходные: {weekend_pm25.mean():.2f}")

t_stat, p_value = stats.ttest_ind(weekday_pm25, weekend_pm25)
print(f"P-value: {p_value:.6f} {'— значимо' if p_value < 0.05 else '— не значимо'}")

# График
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].boxplot([weekday_pm25, weekend_pm25], labels=['Будни', 'Выходные'])
ax[0].set_title('PM2.5: Будни vs Выходные')
ax[0].set_ylabel('PM2.5 (µg/m³)')

ax[1].hist(weekday_pm25, bins=50, alpha=0.6, label='Будни', density=True)
ax[1].hist(weekend_pm25, bins=50, alpha=0.6, label='Выходные', density=True)
ax[1].set_title('Распределение PM2.5')
ax[1].legend()

plt.tight_layout()
plt.savefig('analysis_1_weekday_weekend.png', dpi=300)
print("График 1 сохранён")

# 2. Прогнозирование (на примере PM2.5)
print("\n2. Прогноз PM2.5 по времени")

model_data = summary[['month', 'season', 'hour', 'weekday', 'PM2.5']].dropna()

X = model_data[['month', 'season', 'hour', 'weekday']]
y = model_data['PM2.5']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")

# Важность
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance)

# График
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].scatter(y_test, y_pred, alpha=0.3)
ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax[0].set_xlabel('Реальные')
ax[0].set_ylabel('Предсказанные')
ax[0].set_title(f'Прогноз PM2.5 (R² = {r2:.3f})')

ax[1].barh(importance['feature'], importance['importance'])
ax[1].set_title('Важность признаков')

plt.tight_layout()
plt.savefig('analysis_2_prediction.png', dpi=300)
print("График 2 сохранён")

# 3. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ: SO2 и NO2
print("\n" + "=" * 50)
print("АНАЛИЗ 3: Корреляция между SO2 и NO2")
print("=" * 50)

# Выбираем только нужные колонки и удаляем строки с пропусками в SO2 или NO2
so2_no2_data = summary[['SO2', 'NO2']].dropna()

if len(so2_no2_data) > 0:
    # Корреляция Пирсона
    corr_pearson, p_pearson = stats.pearsonr(so2_no2_data['SO2'], so2_no2_data['NO2'])

    # Корреляция Спирмена
    corr_spearman, p_spearman = stats.spearmanr(so2_no2_data['SO2'], so2_no2_data['NO2'])

    print(f"Корреляция Пирсона: {corr_pearson:.4f} (p-value: {p_pearson:.6f})")
    print(f"Корреляция Спирмена: {corr_spearman:.4f} (p-value: {p_spearman:.6f})")

    if abs(corr_pearson) > 0.7:
        print("✓ Сильная корреляция")
    elif abs(corr_pearson) > 0.3:
        print("○ Средняя корреляция")
    else:
        print("✗ Слабая корреляция")

    # График 1: Scatter SO2 vs NO2
    plt.figure(figsize=(10, 6))
    plt.scatter(so2_no2_data['SO2'], so2_no2_data['NO2'], alpha=0.5, color='teal')
    plt.xlabel('SO2 (ppb)')
    plt.ylabel('NO2 (ppb)')
    plt.title(f'SO2 vs NO2 (корреляция Пирсона = {corr_pearson:.3f})')
    plt.grid(True, alpha=0.3)

    # Линия регрессии
    z = np.polyfit(so2_no2_data['SO2'], so2_no2_data['NO2'], 1)
    p = np.poly1d(z)
    plt.plot(so2_no2_data['SO2'], p(so2_no2_data['SO2']), "r--", alpha=0.8)

    plt.tight_layout()
    plt.savefig('analysis_3_correlation_scatter.png', dpi=300, bbox_inches='tight')

    # График 2: Heatmap корреляции всех загрязнителей
    pollutants = ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
    corr_matrix = summary[pollutants].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
    plt.title('Корреляция между всеми загрязнителями')
    plt.tight_layout()
    plt.savefig('analysis_3_correlation_heatmap.png', dpi=300, bbox_inches='tight')

    print("✓ Графики корреляции сохранены")
else:
    print("Нет данных с одновременными измерениями SO2 и NO2")

# 4. ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ

print("\n" + "=" * 50)
print("АНАЛИЗ 4: Дополнительные исследования")
print("=" * 50)

# 4.1 Тренды по годам
yearly_trends = summary.groupby(summary['Measurement date'].dt.year)[['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']].mean()

plt.figure(figsize=(12, 6))
yearly_trends.plot(marker='o', linewidth=2)
plt.title('Тренды загрязнения воздуха по годам')
plt.ylabel('Средняя концентрация')
plt.xlabel('Год')
plt.legend(title='Вещество')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('analysis_4_trends.png', dpi=300, bbox_inches='tight')
print("✓ График трендов сохранён")

# 4.2 Суточные паттерны
hourly_pattern = summary.groupby('hour')[['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']].mean()

plt.figure(figsize=(12, 6))
hourly_pattern.plot(marker='o', linewidth=2)
plt.title('Суточные паттерны загрязнения')
plt.xlabel('Час дня')
plt.ylabel('Средняя концентрация')
plt.legend(title='Вещество')
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig('analysis_4_hourly.png', dpi=300, bbox_inches='tight')
print("✓ График суточных паттернов сохранён")

# 4.3 Топ-10 самых загрязнённых станций (по PM2.5)
station_pm25 = summary.groupby('Station code')['PM2.5'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
station_pm25.plot(kind='barh', color='coral')
plt.title('Топ-10 станций по среднему PM2.5')
plt.xlabel('Среднее PM2.5 (µg/m³)')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('analysis_4_stations.png', dpi=300, bbox_inches='tight')
print("✓ График станций сохранён")

# Финальный дашборд
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Общий тренд PM2.5 по месяцам
ax1 = fig.add_subplot(gs[0, :])
monthly_pm25 = summary.groupby(summary['Measurement date'].dt.to_period('M'))['PM2.5'].mean()
ax1.plot(monthly_pm25.index.astype(str), monthly_pm25.values, linewidth=2, color='purple')
ax1.set_title('Тренд PM2.5 по месяцам (2017-2019)', fontsize=14, fontweight='bold')
ax1.set_ylabel('PM2.5 (µg/m³)')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Среднее по веществам
ax2 = fig.add_subplot(gs[1, 0])
gas_avg = summary[['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']].mean().sort_values()
gas_avg.plot(kind='barh', ax=ax2, color='skyblue')
ax2.set_title('Средняя концентрация по веществам', fontweight='bold')

# Будни vs Выходные (PM2.5)
ax3 = fig.add_subplot(gs[1, 1])
weekend_pm25 = summary.groupby('is_weekend')['PM2.5'].mean()
ax3.bar(['Будни', 'Выходные'], weekend_pm25.values, color=['#3498db', '#e74c3c'])
ax3.set_title('PM2.5: Будни vs Выходные', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Сезонность
ax4 = fig.add_subplot(gs[1, 2])
season_names = ['Зима', 'Весна', 'Лето', 'Осень']
season_pm25 = summary.groupby('season')['PM2.5'].mean()
ax4.bar(season_names, season_pm25.values, color=['#3498db', '#2ecc71', '#f39c12', '#e67e22'])
ax4.set_title('PM2.5 по сезонам', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Тепловая карта: час × день недели (PM2.5)
ax5 = fig.add_subplot(gs[2, :2])
heatmap_pm25 = summary.pivot_table(values='PM2.5', index='hour', columns='weekday', aggfunc='mean')
sns.heatmap(heatmap_pm25, cmap='YlOrRd', ax=ax5, cbar_kws={'label': 'PM2.5'})
ax5.set_title('PM2.5: Час × День недели', fontweight='bold')
ax5.set_xlabel('День недели (0=Пн)')
ax5.set_ylabel('Час дня')

# Топ станций
ax6 = fig.add_subplot(gs[2, 2])
top5 = summary.groupby('Station code')['PM2.5'].mean().sort_values(ascending=False).head(5)
top5.plot(kind='barh', ax=ax6, color='coral')
ax6.set_title('Топ-5 станций по PM2.5', fontweight='bold')

plt.suptitle('ДАШБОРД: Загрязнение воздуха в Сеуле (2017-2019)', fontsize=18, fontweight='bold', y=0.98)
plt.savefig('final_dashboard.png', dpi=300, bbox_inches='tight')
print("\n✓✓✓ ФИНАЛЬНЫЙ ДАШБОРД СОХРАНЁН: final_dashboard.png ✓✓✓")

print("\n" + "="*60)
print("ВСЕ АНАЛИЗЫ ЗАВЕРШЕНЫ УСПЕШНО!")
print("="*60)