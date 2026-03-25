import os
import sys
import importlib
import torch

# 1. 自動定位 Anomalib 模型路徑
def get_model_class(model_name):
    # 潛在的導入路徑列表
    possible_paths = [
        f"anomalib.models.image.{model_name}.lightning_model",
        f"anomalib.models.{model_name}.lightning_model",
        f"anomalib.models.image.{model_name}",
        f"anomalib.models.{model_name}"
    ]
    
    # 類名映射 (處理大小寫與縮寫不一致)
    class_map = {
        "anomaly_dino": "AnomalyDINO",
        "dinomaly": "Dinomaly",
        "padim": "Padim",
        "efficient_ad": "EfficientAd",
        "csflow": "Csflow",
        "cflow": "Cflow",
        "cfa": "Cfa",
        "dfkde": "Dfkde",
        "dfm": "Dfm",
        "draem": "Draem",
        "dsr": "Dsr",
        "supersimplenet": "Supersimplenet"  # 新增 Supersimplenet
    }
    
    # 獲取目標類別名稱
    target_class = class_map.get(model_name, "".join([p.capitalize() for p in model_name.split("_")]))

    for path in possible_paths:
        try:
            module = importlib.import_module(path)
            model_class = getattr(module, target_class)
            return model_class, path
        except (ImportError, AttributeError):
            continue
    return None, None

def preload():
    print("🚀 啟動 RTX 5070 全模型路徑探測預載 (包含 Supersimplenet)...")
    os.environ["ANOMALIB_SHOW_PROGRESS_BAR"] = "1"

    # 完整模型清單
    target_models = [
        "padim", "efficient_ad", "csflow", "cflow", "cfa", 
        "dfkde", "dfm", "dinomaly", "draem", "dsr", "supersimplenet"
    ]

    success_count = 0
    fail_count = 0

    for name in target_models:
        print(f"\n🔍 探測中: {name}...")
        model_cls, found_path = get_model_class(name)
        
        if model_cls:
            try:
                print(f"✅ 找到路徑: {found_path}")
                print(f"📦 正在初始化 {name} 並同步權重...")
                _ = model_cls()
                print(f"✨ {name.upper()} 已就緒。")
                success_count += 1
            except Exception as e:
                print(f"⚠️ {name} 初始化過程提示: {e}")
                fail_count += 1
        else:
            print(f"❌ 無法在當前路徑中找到 {name}。")
            fail_count += 1

    print("\n" + "="*50)
    print(f"📊 預載總結: 成功 {success_count} 個, 失敗 {fail_count} 個")
    print("="*50)

if __name__ == "__main__":
    preload()