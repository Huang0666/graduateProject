"""
情感标注/审核脚本：交互式标注或审核汽车评论的情感倾向

依赖库：
- pandas==2.1.3

使用方法：
python interactive_labeling.py

输入文件：
- ../../data/raw/all_raw_cars_comments.csv

输出文件：
- ../../data/raw/all_raw_cars_comments_labeled.csv
- ../../data/raw/all_raw_cars_comments_supplement_5000_labeled.csv



标注规则：
0：负面评论
1：正面评论
2：中性评论
3：无关评论

特殊操作：
s: 跳过当前评论 (保留原标注)
q: 退出标注/审核
"""

import pandas as pd
import os
import sys

def load_data(file_path):
    """加载CSV数据文件"""
    if not os.path.exists(file_path):
        print(f"❌ 文件未找到: {file_path}")
        sys.exit(1)
    try:
        # 尝试指定dtype防止混合类型警告，并保留空字符串或数字
        df = pd.read_csv(file_path, keep_default_na=False, dtype={'sentiment_analysis_results': str})
        if 'sentiment_analysis_results' not in df.columns:
            df['sentiment_analysis_results'] = ''
        # 将空字符串转换成NaN以便后续处理，但保留已有标注
        df['sentiment_analysis_results'] = df['sentiment_analysis_results'].replace('', pd.NA)
        return df
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        sys.exit(1)

def save_data(df, file_path):
    """保存标注结果"""
    try:
        # 保存时将NA转换回空字符串，保持CSV格式一致性
        df_save = df.copy()
        df_save['sentiment_analysis_results'] = df_save['sentiment_analysis_results'].fillna('')
        df_save.to_csv(file_path, index=False, encoding='utf-8')
    except Exception as e:
        print(f"❌ 保存数据失败: {e}")

def display_comment(row, current_idx, total_rows):
    """显示评论信息和进度"""
    print("\n" + "="*60)
    # 进度基于总行数
    print(f"[ 进度: {current_idx + 1} / {total_rows} ]")
    print(f"评论ID: {row['id']}")
    print(f"评论内容: {row['content']}")
    current_label = row['sentiment_analysis_results']
    # 检查是否为有效标注 (非NA且非空字符串)
    if pd.notna(current_label) and str(current_label).strip() != '':
        print(f"当前标注: {current_label}")
    else:
        print("当前标注: [未标注]")
    print("="*60)

def get_valid_label():
    """获取有效的标注值 (0-3, s, q)"""
    while True:
        label_input = input("请输入新标注 (0-3) 或 操作 (s:跳过/确认, q:退出): ").strip().lower()
        if label_input == 'q':
            return 'quit'
        if label_input == 's':
            return 'skip' # 跳过或确认当前标注
        try:
            label_int = int(label_input)
            if 0 <= label_int <= 3:
                return label_int # 返回新的标注值
            else:
                print("❌ 无效的标注值，请输入0-3之间的数字")
        except ValueError:
            print("❌ 无效的输入，请输入数字(0-3)或字母(s/q)")

def main():
    script_dir = os.path.dirname(__file__)
    # 修改这里可以修改输入的文件名
    csv_path = os.path.abspath(os.path.join(script_dir, "../../data/raw/all_raw_cars_comments_supplement_5000_1.csv"))
    print(f"ℹ️  目标文件: {csv_path}")

    df = load_data(csv_path)
    total_rows = len(df)

    if total_rows == 0:
        print("❌ 文件为空，无法进行标注。")
        return

    print(f"📝 开始审核/标注，共有 {total_rows} 条评论")

    modified_count = 0
    # 迭代所有行索引
    for i in range(total_rows):
        df_index = df.index[i]
        row = df.loc[df_index]
        display_comment(row, i, total_rows)

        label_action = get_valid_label()

        if label_action == 'quit':
            print("ℹ️  用户选择退出")
            break
        elif label_action == 'skip':
            print("⏭️  跳过/确认当前评论")
            continue # 不修改，直接进入下一条
        else: # 用户输入了新的标注值 0-3
            # 检查新标注是否与旧标注不同 (转换为字符串比较以处理数字和字符串)
            old_label_str = str(df.loc[df_index, 'sentiment_analysis_results']) if pd.notna(df.loc[df_index, 'sentiment_analysis_results']) else ""
            new_label_str = str(label_action)

            if old_label_str != new_label_str:
                df.loc[df_index, 'sentiment_analysis_results'] = label_action
                modified_count += 1
                print(f"✏️  已更新标注为: {label_action}")
                # 保存修改
                save_data(df, csv_path)
            else:
                print("✔️  标注未改变")

    print("\n" + "-"*30)
    print("✨ 审核/标注会话结束")
    if modified_count > 0:
        print(f"本次会话共修改了 {modified_count} 条评论的标注。")
    else:
        print("本次会话未修改任何标注。")
    print("-"*30)

if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("❌ 错误：需要安装 pandas 库才能运行此脚本。")
        print("请在你的虚拟环境中运行: pip install pandas")
        sys.exit(1)
    main() 