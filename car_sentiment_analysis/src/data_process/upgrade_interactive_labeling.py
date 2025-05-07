
"""
情感标注/审核脚本：交互式标注或审核汽车评论的情感倾向

依赖库：
- pandas==2.1.3

使用方法：
python upgrade_interactive_labeling.py


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
import json


def load_data(file_path):
    """加载CSV数据文件"""
    if not os.path.exists(file_path):
        print(f"❌ 文件未找到: {file_path}")
        sys.exit(1)
    try:
        # 读取CSV文件到DataFrame中，指定dtype以避免混合类型警告，使用 | 作为分隔符
        df = pd.read_csv(file_path, sep='|', keep_default_na=False, dtype={'sentiment_analysis_results': str}, encoding='utf-8')
        # 确保DataFrame中存在'sentiment_analysis_results'列
        if 'sentiment_analysis_results' not in df.columns:
            df['sentiment_analysis_results'] = ''
        # 将空字符串替换为NaN以便后续处理
        df['sentiment_analysis_results'] = df['sentiment_analysis_results'].replace('', pd.NA)
        return df
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        sys.exit(1)


def save_data(df, file_path):
    """保存标注结果"""
    try:
        # 创建DataFrame的副本以进行保存
        df_save = df.copy()
        # 将NaN转换回空字符串以保持CSV格式一致性
        df_save['sentiment_analysis_results'] = df_save['sentiment_analysis_results'].fillna('')
        # 保存DataFrame到CSV文件，使用 | 作为分隔符
        df_save.to_csv(file_path, sep='|', index=False, encoding='utf-8')
    except Exception as e:
        print(f"❌ 保存数据失败: {e}")


def save_last_position(file_path, position):
    """保存最后标注的位置"""
    progress_file = os.path.join(os.path.dirname(file_path), '.labeling_progress.json')
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({'last_position': position, 'file': os.path.basename(file_path)}, f)
    except Exception as e:
        print(f"⚠️ 保存进度信息失败: {e}")


def load_last_position(file_path):
    """加载上次标注的位置"""
    progress_file = os.path.join(os.path.dirname(file_path), '.labeling_progress.json')
    try:
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get('file') == os.path.basename(file_path):
                    return data.get('last_position')
    except Exception as e:
        print(f"⚠️ 读取进度信息失败: {e}")
    return None


def display_comments(rows, start_idx, total_rows):
    """显示多条评论信息和进度"""
    print("\n" + "="*60)
    for i, (_, row) in enumerate(rows.iterrows()):
        # 显示当前处理的评论进度
        print(f"[ 进度: {start_idx + i + 1} / {total_rows} ]")
        # 显示评论ID和内容
        print(f"评论ID: {row.id}")
        print(f"评论内容: {row.content}")
        # 显示当前标注状态
        current_label = row.sentiment_analysis_results
        if pd.notna(current_label) and str(current_label).strip() != '':
            print(f"当前标注: {current_label}")
        else:
            print("当前标注: [未标注]")
        print("-"*60)


def get_valid_labels(batch_size):
    """获取有效的标注值 (0-3, s, q)"""
    while True:
        # 提示用户输入标注值或操作命令
        labels_input = input(f"请输入新标注 (0-3) 或 操作 (s:跳过/确认, q:退出) 共{batch_size}个: ").strip().lower()
        if labels_input == 'q':
            return 'quit'
        if labels_input == 's':
            return 'skip'
        # 将输入的标注值分割成列表
        labels = labels_input.split()
        if len(labels) != batch_size:
            print(f"❌ 请输入{batch_size}个标注值")
            continue
        try:
            # 尝试将输入转换为整数列表
            labels_int = [int(label) for label in labels]
            # 检查所有标注值是否在有效范围内
            if all(0 <= label <= 3 for label in labels_int):
                return labels_int
            else:
                print("❌ 无效的标注值，请输入0-3之间的数字")
        except ValueError:
            print("❌ 无效的输入，请输入数字(0-3)或字母(s/q)")


def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(__file__)
    # 设置输入CSV文件的绝对路径
    csv_path = os.path.abspath(os.path.join(script_dir, "../../data/raw/all_raw_cars_comments_supplement_10000.csv"))
    print(f"ℹ️  目标文件: {csv_path}")

    # 加载评论数据
    df = load_data(csv_path)
    total_rows = len(df)

    if total_rows == 0:
        print("❌ 文件为空，无法进行标注。")
        return

    # 检查是否存在上次的标注位置
    last_position = load_last_position(csv_path)
    start_idx = 0
    
    if last_position is not None:
        while True:
            choice = input(f"发现上次标注到第 {last_position + 1} 条评论，是否从该位置继续？(y/n): ").strip().lower()
            if choice == 'y':
                start_idx = last_position
                break
            elif choice == 'n':
                break
            else:
                print("❌ 请输入 y 或 n")

    print(f"📝 开始审核/标注，共有 {total_rows} 条评论")

    # 记录本次会话中被修改的评论数量
    modified_count = 0
    # 每次批量处理的评论数量
    batch_size = 5
    # 迭代处理每批评论
    for current_idx in range(start_idx, total_rows, batch_size):
        # 计算当前批次的结束索引
        end_idx = min(current_idx + batch_size, total_rows)
        # 获取当前批次的评论
        rows = df.iloc[current_idx:end_idx]
        # 显示当前批次的评论信息
        display_comments(rows, current_idx, total_rows)

        # 获取用户输入的标注值
        label_action = get_valid_labels(len(rows))

        if label_action == 'quit':
            print("ℹ️  用户选择退出")
            # 保存当前位置
            save_last_position(csv_path, current_idx)
            break
        elif label_action == 'skip':
            print("⏭️  跳过/确认当前评论")
            continue
        else:
            # 更新标注结果
            for i, label in enumerate(label_action):
                df_index = df.index[current_idx + i]
                old_label_str = str(df.loc[df_index, 'sentiment_analysis_results']) if pd.notna(df.loc[df_index, 'sentiment_analysis_results']) else ""
                new_label_str = str(label)

                if old_label_str != new_label_str:
                    df.loc[df_index, 'sentiment_analysis_results'] = label
                    modified_count += 1
                    print(f"✏️  已更新标注为: {label}")
            # 保存修改后的数据
            save_data(df, csv_path)
            # 保存当前位置
            save_last_position(csv_path, current_idx)

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