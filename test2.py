import csv
import datetime
import random

# 参数配置
n = 50  # 数据条数（可修改）
start_time = datetime.datetime(2024, 1, 1, 8, 0)  # 起始时间
interval = datetime.timedelta(hours=1)  # 时间间隔（每小时生成一条数据）
heart_rate_range = (50, 120)  # 心率随机范围（正常范围）

# 生成数据
data = []
current_time = start_time
for _ in range(n):
    timestamp = current_time.strftime("%Y-%m-%d %H:%M")
    heart_rate = random.randint(heart_rate_range[0], heart_rate_range[1])
    data.append([timestamp, heart_rate])
    current_time += interval

# 写入 CSV 文件
with open("health_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "heart_rate"])  # 表头
    writer.writerows(data)
