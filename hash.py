import hashlib
import os


def calculate_md5(file_path):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def remove_duplicates(dataset_dir):
    print(f"开始扫描重复图片: {dataset_dir} ...")
    total_removed = 0

    # 遍历每个类别文件夹
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        hashes = {}
        duplicates_in_class = 0

        for filename in os.listdir(class_path):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                continue

            file_path = os.path.join(class_path, filename)
            try:
                file_hash = calculate_md5(file_path)

                if file_hash in hashes:
                    # 发现重复，删除
                    print(f"[删除] {class_name}/{filename} (与 {hashes[file_hash]} 重复)")
                    os.remove(file_path)
                    duplicates_in_class += 1
                    total_removed += 1
                else:
                    hashes[file_hash] = filename
            except Exception as e:
                print(f"无法读取文件 {file_path}: {e}")

        if duplicates_in_class > 0:
            print(f"--> 类别 '{class_name}' 清理了 {duplicates_in_class} 张重复图")

    print(f"\n清理完成！总共删除了 {total_removed} 张重复图片。")


if __name__ == '__main__':
    # 修改为你的数据集路径
    DATA_DIR = './datasets'
    if os.path.exists(DATA_DIR):
        remove_duplicates(DATA_DIR)
    else:
        print("数据集目录不存在")