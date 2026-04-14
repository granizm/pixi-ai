.PHONY: init download-models build run clean

# 初期セットアップ（初回・環境再構築時に実行）
init: download-models
	@echo "==> Init complete."

# モデルファイルのダウンロード
download-models:
	@echo "==> Downloading ML models..."
	sh scripts/setup.sh download-models

# ビルド（開発用）
build:
	cargo build --release

# 実行
run:
	cargo run --release

# クリーン
clean:
	cargo clean
