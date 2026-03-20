Bạn dựa vào các file này: logic, luồng xử lý,...
    run_swat.py
    data
        + SWAT.py
    configs
        + classification
            + carla_classification_swat.yml
        + pretext
            + carla_pretext_swat.yml

Bạn tinh chỉnh file run_wadi.py và các file liên quan cho phù hợp để tui chạy file run_wadi.py trên kaggle

THÔNG TIN DATASET (RẤT QUAN TRỌNG)
========================
Train:
- Path: /kaggle/input/datasets/giovannimonco/wadi-data/WADI_14days_new.csv
- Chỉ chứa dữ liệu normal (không có attack)
- Có các cột: Row, Date, Time + các sensor
- KHÔNG có label

Test:
- Path: /kaggle/input/datasets/giovannimonco/wadi-data/WADI_attackdataLABLE.csv
- Row 0: 0,1,2,3,...(header giả do export lỗi)
- Row 1: Row, Date, Time, ... (header thật)
- Row 2+: data
- Cột cuối là label: "Attack LABLE (1:No Attack, -1:Attack)"

Mapping label:
- 1 → 0 (normal)
- -1 → 1 (anomaly)