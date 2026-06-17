import csv
from config import ITEM_META_LLM
import pandas as pd

rows = []
bad_lines_info = []
with open(ITEM_META_LLM, 'r', encoding='gbk') as f:
    reader = csv.reader(f)
    header = next(reader)
    for line_num, row in enumerate(reader, start=2):  # 第1行是表头，数据从第2行开始
        try:
            item_id = int(row[0])
            rows.append(row)
        except (ValueError, IndexError):
            bad_lines_info.append((line_num, row))

print(f"成功读取 {len(rows)} 行，跳过脏数据 {len(bad_lines_info)} 行")

with open("bad_lines_report.txt", "w", encoding="utf-8") as f:
    f.write(f"共发现 {len(bad_lines_info)} 条脏数据\n\n")
    for line_num, row in bad_lines_info:
        f.write(f"行号: {line_num}\n内容: {row}\n{'-'*50}\n")
print("脏数据报告已写入 bad_lines_report.txt")

# 构造 DataFrame 并保存为干净 CSV
df_clean = pd.DataFrame(rows, columns=header)
df_clean['item_id'] = df_clean['item_id'].astype(int)

clean_path = ITEM_META_LLM.replace('.csv', '_clean.csv')
df_clean.to_csv(clean_path, index=False, encoding='utf-8-sig', quoting=1)  # quoting=1 防止后续引号问题
print(f"干净数据已保存至: {clean_path}")